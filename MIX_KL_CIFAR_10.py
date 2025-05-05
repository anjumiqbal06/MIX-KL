import argparse
import copy
import logging
import os
import time
import numpy as np
import torch
from torch.distributions.beta import Beta
from CIFAR10_models import *
import torch.nn as nn
from utils import *
from Feature_model.feature_resnet import *
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt  

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='CIFAR10', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    parser.add_argument('--normal_mean', default=0, type=float)
    parser.add_argument('--normal_std', default=1, type=float)
    parser.add_argument('--out_dir', default='training_output', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lamda', default=12, type=float)
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--factor', default=0.7, type=float)
    parser.add_argument('--mixup-alpha', default=1.0, type=float)
    parser.add_argument('--EMA_value', default=0.82, type=float)
    parser.add_argument('--temperature', default=1.0, type=float, help="Temperature scaling factor")
    return parser.parse_args()

args = get_args()

class Mixup:
    def __init__(self, alpha=1.0):
        self.beta = Beta(alpha, alpha)
        
    def __call__(self, x, y):
        lam = self.beta.sample().item()
        index = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_onehot = F.one_hot(y, num_classes=10).float()
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]
        return mixed_x, mixed_y

def get_loaders_mixup(dir_, batch_size):
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
    train_dataset = datasets.CIFAR10(dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(dir_, train=False, transform=test_transform, download=True)
    
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader

def _label_smoothing(label, factor):
    if len(label.shape) == 1:
        one_hot = F.one_hot(label, num_classes=10).float().cuda()
    else:
        one_hot = label
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / (10 - 1))
    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
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

class EvaluationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        output, _ = self.model(x)
        return output

def main():
    args = get_args()
    output_path = os.path.join(args.out_dir, 'cifar10')
    output_path = os.path.join(output_path, 'epsilon_' + str(args.epsilon))
    output_path = os.path.join(output_path, 'alpha_' + str(args.alpha))
    output_path = os.path.join(output_path, 'model_' + str(args.model))
    output_path = os.path.join(output_path, 'factor_' + str(args.factor))
    output_path = os.path.join(output_path, 'mixup_alpha_' + str(args.mixup_alpha))
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

    train_loader, test_loader = get_loaders_mixup(args.data_dir, args.batch_size)
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

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * 99 / 110, lr_steps * 104 / 110],
                                                         gamma=0.1)

    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []

    # Adding lists to store accuracy values for plotting
    clean_acc_list = []
    pgd_acc_list = []
    
    for epoch in range(args.epochs):
        epoch_time = 0
        train_loss = 0
        train_acc = 0
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

            adv_output, ori_fea_output = model(mixed_X + delta)
            adv_output = torch.nn.Softmax(dim=1)(adv_output)
            ori_fea_output = torch.nn.Softmax(dim=1)(ori_fea_output)

            # Replacing cross entropy loss with KL Divergence with temperature scaling
            log_adv_output = F.log_softmax(adv_output / args.temperature, dim=1)
            loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
            loss = loss_fn(log_adv_output, mixed_y)  # KL Divergence Loss with Temperature Scaling

            adv_output, ori_fea_output = model(mixed_X)
            adv_output = torch.nn.Softmax(dim=1)(adv_output)
            ori_fea_output = torch.nn.Softmax(dim=1)(ori_fea_output)

            loss.backward(retain_graph=True)
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - mixed_X, upper_limit - mixed_X)
            delta = delta.detach()
            
            ori_output, fea_output = model(mixed_X + delta)
            output = torch.nn.Softmax(dim=1)(ori_output)
            fea_output = torch.nn.Softmax(dim=1)(fea_output)

            label_smoothing = _label_smoothing(mixed_y.argmax(1), args.factor)
            
            loss = LabelSmoothLoss(ori_output, label_smoothing) + args.lamda * (
                loss_fn(output, adv_output) + loss_fn(fea_output, ori_fea_output)) / (
                loss_fn(mixed_X + delta, mixed_X) + 0.125)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (ori_output.argmax(1) == mixed_y.argmax(1)).sum().item()
            train_n += y.size(0)
            
            clean_acc = (adv_output.argmax(1) == mixed_y.argmax(1)).sum().item()
            adv_acc = (output.argmax(1) == mixed_y.argmax(1)).sum().item()
            if adv_acc / clean_acc < args.EMA_value:
                teacher_model.update_params(model)
                teacher_model.apply_shadow()

            scheduler.step()
            epoch_time += time.time() - batch_start_time

        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time, scheduler.get_last_lr()[0], 
                    train_loss / train_n, train_acc / train_n)

        # Use EvaluationWrapper for testing
        model_test = EvaluationWrapper(copy.deepcopy(teacher_model.model)).eval()
        eval_epsilon = (args.epsilon / 255.) / std
        pgd_loss, pgd_acc = evaluate_powerful_pgd(test_loader, model_test, 10, 1, eval_epsilon)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)

        clean_acc_list.append(test_acc)
        pgd_acc_list.append(pgd_acc)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        
        if pgd_acc > best_result:
            best_result = pgd_acc
            # Save original model state dict
            torch.save(teacher_model.model.state_dict(), os.path.join(output_path, 'best_model.pth'))

        # Save plot at every epoch
        plt.figure(figsize=(10, 5))
        plt.plot(range(epoch + 1), clean_acc_list, label='Clean Accuracy')
        plt.plot(range(epoch + 1), pgd_acc_list, label='PGD Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.title(f'Epoch {epoch+1} Accuracy Plot')
        plt.savefig(os.path.join(output_path, f'epoch_{epoch+1}_accuracy_plot.png'))


    # Save final model
    torch.save(teacher_model.model.state_dict(), os.path.join(output_path, 'final_model.pth'))
    logger.info("Clean Accuracies: %s", str(epoch_clean_list))
    logger.info("PGD Accuracies: %s", str(epoch_pgd_list))

if __name__ == "__main__":
    main()
