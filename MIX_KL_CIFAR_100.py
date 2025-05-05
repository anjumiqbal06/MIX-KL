import argparse
import copy
import logging
import os
import time
import numpy as np
import torch
from torch.distributions.beta import Beta
from Cifar100_models import *
from utils import *
from Feature_model.feature_resnet_cifar100 import *
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import matplotlib.pyplot as plt  

logger = logging.getLogger(__name__)

class Mixup:
    def __init__(self, alpha=1.0):
        self.beta = Beta(alpha, alpha)
        
    def __call__(self, x, y):
        lam = self.beta.sample().item()
        index = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_onehot = F.one_hot(y, num_classes=100).float()  
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]
        return mixed_x, mixed_y

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='CIFAR100', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    parser.add_argument('--out_dir', default='training_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--lamda', default=42, type=float, help='Label Smoothing')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--EMA_value', default=0.55, type=float)
    parser.add_argument('--mixup-alpha', default=1.0, type=float, help="Mixup alpha value")
    parser.add_argument('--temperature', default=1.0, type=float, help="Temperature scaling factor")
    return parser.parse_args()

args = get_args()

def get_loaders_cifar100_mixup(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    num_workers = 0
    train_dataset = datasets.CIFAR100(dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(dir_, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader

def _label_smoothing(label, factor):
    one_hot = np.eye(100)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(100 - 1))
    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

class EMA(object):
    def __init__(self, model, alpha=0.9998, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

def kl_divergence_loss_with_temperature(predicted, target, temperature):
    predicted = F.softmax(predicted / temperature, dim=-1)
    target = F.softmax(target / temperature, dim=-1)
    kl_div = F.kl_div(F.log_softmax(predicted, dim=-1), target, reduction='batchmean')
    return kl_div

def combined_loss(adv_output, ori_fea_output, mixed_y, temperature, lamda):
    ce_loss = F.cross_entropy(adv_output, mixed_y.argmax(1))
    # KL Divergence Loss with temperature scaling
    kl_loss = kl_divergence_loss_with_temperature(adv_output, mixed_y, temperature)
    # Combined loss
    total_loss = ce_loss + lamda * kl_loss
    return total_loss

def main():
    args = get_args()
    output_path = os.path.join(args.out_dir, 'cifar100')
    output_path = os.path.join(output_path, 'factor_' + str(args.factor))
    output_path = os.path.join(output_path, 'EMA_value_' + str(args.EMA_value))
    output_path = os.path.join(output_path, 'lamda_' + str(args.lamda))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logfile = os.path.join(output_path, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(output_path, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders_cifar100_mixup(args.data_dir, args.batch_size)
    mixup = Mixup(alpha=args.mixup_alpha)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    print('==> Building model..')
    logger.info('==> Building model..')
    if args.model == "VGG":
        model = VGG('VGG19')
    elif args.model == "ResNet18":
        model = Feature_ResNet18()
    elif args.model == "PreActResNest18":
        model = PreActResNet18()
    elif args.model == "WideResNet":
        model = WideResNet()
    model = model.cuda()
    model.train()
    teacher_model = EMA(model)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    lr_up = 20
    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps =  len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.0, max_lr=args.lr_max,
                                                      step_size_up=lr_steps * lr_up,
                                                      step_size_down=lr_steps * (args.epochs - lr_up))
    # Training
    prev_robust_acc = 0.
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_result = 0
    epoch_train_clean_list = []
    epoch_train_pgd_list = []
    epoch_clean_list = []
    epoch_pgd_list = []
    init_loss = []
    init_acc = []
    final_loss = []
    final_acc = []
    for epoch in range(args.epochs):
        epoch_time = 0
        train_loss = 0
        train_acc = 0
        init_train_loss = 0
        init_train_acc = 0
        train_n = 0
        teacher_model.model.eval()
        for i, (X, y) in enumerate(train_loader):
            batch_start_time = time.time()
            X, y = X.cuda(), y.cuda()

            mixed_X, mixed_y = mixup(X, y)

            if args.delta_init != 'previous':
                delta = torch.zeros_like(mixed_X).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - mixed_X, upper_limit - mixed_X)

            delta = epsilon/2 * torch.sign(delta)
            delta.data = clamp(delta, lower_limit - mixed_X, upper_limit - mixed_X)
            delta.requires_grad = True

            # Save the original delta as temp_delta for comparison
            temp_delta = delta.clone().detach()

            adv_output, ori_fea_output = model(mixed_X + delta)
            adv_output = torch.nn.Softmax(dim=1)(adv_output)
            ori_fea_output = torch.nn.Softmax(dim=1)(ori_fea_output)

            # Using the combined loss function
            loss = combined_loss(adv_output, ori_fea_output, mixed_y, args.temperature, args.lamda)

            init_train_loss += loss.item() * y.size(0)
            init_train_acc += (adv_output.max(1)[1] == y).sum().item()

            loss.backward(retain_graph=True)
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - mixed_X, upper_limit - mixed_X)
            delta = delta.detach()

            ori_output, fea_output = model(mixed_X + delta)
            output = torch.nn.Softmax(dim=1)(ori_output)
            fea_output = torch.nn.Softmax(dim=1)(fea_output)

            loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
            label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.factor)).cuda())

            loss = LabelSmoothLoss(ori_output, label_smoothing.float()) + args.lamda * (loss_fn(output.float(), adv_output.float()) + loss_fn(fea_output.float(), ori_fea_output.float())) / (loss_fn((mixed_X + delta).float(), (mixed_X + temp_delta).float()) + 0.125)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (ori_output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            adv_acc = (output.max(1)[1] == mixed_y.argmax(1)).sum().item()
            clean_acc = (adv_output.max(1)[1] == mixed_y.argmax(1)).sum().item()

            if adv_acc / (clean_acc + 1) < args.EMA_value:
                teacher_model.update_params(model)
                teacher_model.apply_shadow()

            scheduler.step()
            batch_end_time = time.time()
            epoch_time += batch_end_time - batch_start_time

        init_loss.append(init_train_loss / train_n)
        init_acc.append(init_train_acc / train_n)
        final_loss.append(train_loss / train_n)
        final_acc.append(train_acc / train_n)

        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time, lr, train_loss / train_n, train_acc / train_n)

        logger.info('==> Building model..')
        if args.model == "VGG":
            model_test = VGG('VGG19').cuda()
        elif args.model == "ResNet18":
            model_test = ResNet18().cuda()
        elif args.model == "PreActResNest18":
            model_test = PreActResNet18().cuda()
        elif args.model == "WideResNet":
            model_test = WideResNet().cuda()

        model_test.load_state_dict(teacher_model.model.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)

        if best_result <= pgd_acc:
            best_result = pgd_acc
            torch.save(model_test.state_dict(), os.path.join(output_path, 'best_model.pth'))
            
    torch.save(model_test.state_dict(), os.path.join(output_path, 'final_model.pth'))
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    print(epoch_clean_list)
    print(epoch_pgd_list)

if __name__ == "__main__":
    main()
