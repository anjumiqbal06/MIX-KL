import argparse
import copy
import logging
import os
import time
import numpy as np
import torch
from torch.distributions.beta import Beta
import torch.nn as nn
from utils01 import *
from Feature_model.feature_preact_resnet import *
from ImageNet_models import *
from TinyImageNet import TinyImageNet
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
logger = logging.getLogger(__name__)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mean_tensor = torch.tensor(mean, device=device).view(3, 1, 1)
std_tensor = torch.tensor(std, device=device).view(3, 1, 1)
lower_limit = (0.0 - mean_tensor) / std_tensor
upper_limit = (1.0 - mean_tensor) / std_tensor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--data-dir', default='E:/MIX-KL/FGSM-LAW-main/data/tiny-imagenet-200', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model', default='PreActResNest18', type=str, help='model name')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    parser.add_argument('--normal-mean', default=0, type=float)
    parser.add_argument('--normal-std', default=1, type=float)
    parser.add_argument('--out-dir', default='train_fgsm_RS_output', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lamda', default=42.0, type=float)
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--mixup-alpha', default=1.0, type=float)
    parser.add_argument('--ema-value', default=0.55, type=float)
    parser.add_argument('--tau0', default=2.0, type=float, help='Initial temperature for distillation')
    parser.add_argument('--decay-gamma', default=0.99, type=float, help='Geometric decay factor for temperature')
    parser.add_argument('--lambda-kl', default=1.0, type=float, help='Weight for KL divergence loss')
    parser.add_argument('--lambda-kl-base', default=1.0, type=float, help='Base weight for KL divergence loss')
    parser.add_argument('--c-num', default=0.125, type=float)
    parser.add_argument('--length', default=100, type=int)
    return parser.parse_args()

args = get_args()

class Mixup:
    def __init__(self, alpha=1.0):
        self.beta = Beta(alpha, alpha)
        
    def __call__(self, x, y):
        lam = self.beta.sample().item()
        index = torch.randperm(x.size(0)).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_onehot = F.one_hot(y, num_classes=200).float()  # 200 classes for Tiny ImageNet
        mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]
        return mixed_x, mixed_y

def convert_to_rgb(img):
    return img.convert('RGB')

def imagenet_loaders_64(root: str, batch_size: int):
    transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = TinyImageNet(root, split='train', transform=transform, in_memory=True)
    val_set = TinyImageNet(root, split='val', transform=transform, in_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )
    return train_loader, val_loader

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

def _label_smoothing(label: torch.Tensor, factor: float):
    one_hot = torch.eye(200, device=label.device)[label]
    result = one_hot * factor + (one_hot - 1.0) * ((factor - 1.0) / (one_hot.size(1) - 1))
    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

def geometric_temperature_scaling(logits, temperature):
    """Apply geometric temperature scaling to logits"""
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    scaled_probs = torch.pow(probs, 1/temperature)
    scaled_probs = scaled_probs / scaled_probs.sum(dim=1, keepdim=True)
    return scaled_probs

def kl_div_geometric_temperature(student_logits, teacher_logits, temperature):
    """KL divergence with geometric temperature scaling"""
    student_geo = geometric_temperature_scaling(student_logits, temperature)
    teacher_geo = geometric_temperature_scaling(teacher_logits, temperature)
    
    student_log_geo = torch.log(student_geo + 1e-8)
    kl = F.kl_div(student_log_geo, teacher_geo, reduction='batchmean')
    
    return kl * (temperature ** 2)

def main():
    args = get_args()
    
    # Create output directory with parameter details
    out_dir = os.path.join(
        args.out_dir,
        'Tiny_ImageNet',
        f'c_num_{args.c_num}',
        f'model_{args.model}',
        f'factor_{args.factor}',
        f'length_{args.length}',
        f'EMA_{args.ema_value}',
        f'lamda_{args.lamda}',
        f'tau0_{args.tau0}',
        f'gamma_{args.decay_gamma}',
        f'lambda_kl_{args.lambda_kl}',
        f'lambda_kl_base_{args.lambda_kl_base}',
    )
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(out_dir, 'output.log')
    )
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, val_loader = imagenet_loaders_64(args.data_dir, args.batch_size)
    mixup = Mixup(alpha=args.mixup_alpha)

    epsilon = (args.epsilon / 255.0)
    alpha_pixel = (args.alpha / 255.0) / std_tensor
    alpha_pixel = alpha_pixel.view(1, 3, 1, 1)

    print('==> Building model..')
    logger.info('==> Building model..')
    if args.model == 'VGG':
        model = VGG('VGG19')
    elif args.model == 'PreActResNest18':
        model = Feature_PreActResNet18()
    elif args.model == 'ResNet18':
        model = ResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet()
    else:
        raise ValueError(f"Unknown model {args.model}")
        
    model = model.to(device)
    model.train()
    teacher_model = EMA(model, alpha=args.ema_value)
    
    optim = torch.optim.SGD(model.parameters(), lr=args.lr_max,
                            momentum=args.momentum, weight_decay=args.weight_decay)

    lr_up_epochs = 20
    steps_per_epoch = len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optim, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=steps_per_epoch // 2,
            step_size_down=steps_per_epoch // 2,
        )
    else: 
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optim, base_lr=0.0, max_lr=args.lr_max,
            step_size_up=steps_per_epoch * lr_up_epochs,
            step_size_down=steps_per_epoch * (args.epochs - lr_up_epochs),
        )

    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t Current Temp \t KL Weight')
    best_pgd_acc = 0.0
    clean_acc_trend, pgd_acc_trend = [], []
    temperature_history = []
    kl_weight_history = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        teacher_model.model.eval()
        
        current_tau = args.tau0 * (args.decay_gamma ** epoch)
        temperature_history.append(current_tau)
        
        kl_weight = args.lambda_kl_base * (current_tau / args.tau0) ** 2
        kl_weight_history.append(kl_weight)
        
        running_loss, running_acc, running_n = 0.0, 0, 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            mixed_X, mixed_y = mixup(X, y)


            if args.delta_init == 'previous':
                if 'delta' not in locals():
                    delta = torch.zeros_like(mixed_X)
            else:
                delta = torch.zeros_like(mixed_X)
                if args.delta_init == 'random':
                    delta.uniform_(-epsilon, epsilon)
            delta = (epsilon / 2) * delta.sign()
            delta.data = clamp(delta, lower_limit - mixed_X, upper_limit - mixed_X)
            delta.requires_grad_(True)

            temp_delta = delta.detach().clone()

            adv_out, ori_fea_out = model(mixed_X + delta)
            loss = LabelSmoothLoss(adv_out, mixed_y)
            loss.backward(retain_graph=True)
            
            grad = F.interpolate(delta.grad.detach(), size=(64, 64), mode='bilinear', align_corners=False)
            delta.data = clamp(delta + alpha_pixel * grad.sign(), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - mixed_X, upper_limit - mixed_X)
            delta = delta.detach()

            optim.zero_grad()

            ori_out, fea_out = model(mixed_X + delta)
            clean_out, _ = model(mixed_X)

            with torch.no_grad():
                teacher_out, _ = teacher_model.model(mixed_X + delta)

            kl_loss = kl_div_geometric_temperature(ori_out, teacher_out, current_tau)

            ori_out_soft = F.softmax(ori_out, dim=1)
            adv_out_soft = F.softmax(clean_out, dim=1)
            fea_out_soft = F.softmax(fea_out, dim=1)
            ori_fea_out_soft = F.softmax(ori_fea_out, dim=1)

            mse = nn.MSELoss()
            consistency_loss = (mse(ori_out_soft, adv_out_soft) + mse(fea_out_soft, ori_fea_out_soft)) / (
                mse(mixed_X + delta, mixed_X + temp_delta) + args.c_num)

            label_loss = LabelSmoothLoss(ori_out, mixed_y)
            total_loss = label_loss + args.lamda * consistency_loss + kl_weight * kl_loss

            total_loss.backward()
            optim.step()

            running_loss += total_loss.item() * y.size(0)
            running_acc += (ori_out.argmax(1) == y).sum().item()
            running_n += y.size(0)

            clean_acc = (clean_out.argmax(1) == y).sum().item()
            adv_acc = (ori_out.argmax(1) == y).sum().item()
            if adv_acc / (clean_acc + 1e-8) < args.ema_value:
                teacher_model.update_params(model)
                teacher_model.apply_shadow()

            scheduler.step()

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / running_n
        train_acc = running_acc / running_n
        lr_now = scheduler.get_last_lr()[0]
        
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time, lr_now, train_loss, train_acc, current_tau, kl_weight)

        model_test = EvaluationWrapper(copy.deepcopy(teacher_model.model)).to(device)
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(val_loader, model_test, 10, 1)
        val_loss, val_acc = evaluate_standard(val_loader, model_test)
        clean_acc_trend.append(val_acc)
        pgd_acc_trend.append(pgd_acc)
        
        logger.info('ValLoss %.4f\tValAcc %.4f\tPGDLoss %.4f\tPGDAcc %.4f', 
                   val_loss, val_acc, pgd_loss, pgd_acc)

        if pgd_acc >= best_pgd_acc:
            best_pgd_acc = pgd_acc
            torch.save(model_test.state_dict(), os.path.join(out_dir, 'best_model.pth'))

    torch.save(model_test.state_dict(), os.path.join(out_dir, 'final_model.pth'))
    logger.info('Training complete')
    logger.info('Clean Accuracies: %s', str(clean_acc_trend))
    logger.info('PGD Accuracies: %s', str(pgd_acc_trend))
    logger.info('Temperature History: %s', str(temperature_history))
    logger.info('KL Weight History: %s', str(kl_weight_history))


if __name__ == "__main__":
    main()
