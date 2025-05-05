import argparse
import copy
import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from torchvision import datasets, transforms
from torch.distributions.beta import Beta
from tqdm.auto import tqdm
from utils01 import *  
from Feature_model.feature_preact_resnet import *
from ImageNet_models import *
from TinyImageNet import TinyImageNet

logger = logging.getLogger(__name__)

class Mixup:
    def __init__(self, alpha: float = 1.0):
        self.beta = Beta(alpha, alpha)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        lam = self.beta.sample().item()
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[index]
        y_onehot = F.one_hot(y, num_classes=200).float()
        mixed_y = lam * y_onehot + (1.0 - lam) * y_onehot[index]
        return mixed_x, mixed_y

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--data-dir', default='E:/MIX-KL/data/tiny-imagenet-200', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0.0, type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model', default='PreActResNest18', type=str, help='model name')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size (pixel space)')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    parser.add_argument('--normal-mean', default=0.0, type=float)
    parser.add_argument('--normal-std', default=1.0, type=float)
    parser.add_argument('--out-dir', default='training_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lamda', default=42.0, type=float, help='Label-smoothing lambda for KL term')
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--ema-value', dest='ema_value', default=0.55, type=float)  # follow PEP-8 naming
    parser.add_argument('--mixup-alpha', default=1.0, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--c-num', default=0.125, type=float)
    parser.add_argument('--length', default=100, type=int)
    return parser.parse_args()

args = get_args() 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mean_tensor = torch.tensor(mean, device=device).view(3, 1, 1)
std_tensor = torch.tensor(std, device=device).view(3, 1, 1)

lower_limit = (0.0 - mean_tensor) / std_tensor
upper_limit = (1.0 - mean_tensor) / std_tensor

print(f"[DEBUG] lower_limit {lower_limit.shape} upper_limit {upper_limit.shape}")

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

def _label_smoothing(label: torch.Tensor, factor: float):
    one_hot = torch.eye(200, device=label.device)[label]
    result = one_hot * factor + (one_hot - 1.0) * ((factor - 1.0) / (one_hot.size(1) - 1))
    return result

def label_smooth_loss(inputs: torch.Tensor, targets: torch.Tensor):
    logp = F.log_softmax(inputs, dim=-1)
    return (-targets * logp).sum(dim=-1).mean()

class EMA:
    def __init__(self, model: nn.Module, alpha: float = 0.9998, buffer_ema: bool = True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self._get_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def _get_state(self):
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    def update(self, model: nn.Module):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        current = model.state_dict()
       
        for k in self.param_keys:
            self.shadow[k].mul_(decay).add_(current[k], alpha=1 - decay)
        if self.buffer_ema:
            for k in self.buffer_keys:
                if self.shadow[k].is_floating_point():
                    self.shadow[k].mul_(decay).add_(current[k], alpha=1 - decay)
                else:
                    self.shadow[k].copy_(current[k])    
        self.step += 1

    def apply_shadow(self):
        self.backup = self._get_state()
        self.model.load_state_dict(self.shadow, strict=True)

    def restore(self):
        self.model.load_state_dict(self.backup, strict=True)

def kl_div_with_temp(predict: torch.Tensor, target: torch.Tensor, t: float):
    p = F.softmax(predict / t, dim=-1)
    q = F.softmax(target / t, dim=-1)
    return F.kl_div(F.log_softmax(p, dim=-1), q, reduction='batchmean')

def combined_loss(adv_out, ori_fea_out, mixed_y, temp, lam):
    ce = F.cross_entropy(adv_out, mixed_y.argmax(1))
    kl = kl_div_with_temp(adv_out, mixed_y, temp)
    return ce + lam * kl

def main():
    out_dir = os.path.join(
        args.out_dir,
        'Tiny_ImageNet',
        f'c_num_{args.c_num}',
        f'model_{args.model}',
        f'factor_{args.factor}',
        f'length_{args.length}',
        f'EMA_{args.ema_value}',
        f'lamda_{args.lamda}',
    )
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(out_dir, 'output.log'),
    )
    logger.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_loader, val_loader = imagenet_loaders_64(args.data_dir, args.batch_size)
    mixup = Mixup(alpha=args.mixup_alpha)
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

    model = model.to(device).train()
    teacher = EMA(model)

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
    else:  # multistep
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optim, base_lr=0.0, max_lr=args.lr_max,
            step_size_up=steps_per_epoch * lr_up_epochs,
            step_size_down=steps_per_epoch * (args.epochs - lr_up_epochs),
        )
    epsilon = args.epsilon / 255.0
    alpha_pixel = (args.alpha / 255.0) / std_tensor  # per-channel step
    alpha_pixel = alpha_pixel.view(1, 3, 1, 1)
    best_pgd_acc = 0.0
    clean_acc_trend, pgd_acc_trend = [], []
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        running_loss, running_acc, running_n = 0.0, 0, 0
        init_loss, init_acc = 0.0, 0

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
            adv_out_soft = F.softmax(adv_out, dim=1)
            ori_fea_out_soft = F.softmax(ori_fea_out, dim=1)

            loss = combined_loss(adv_out, ori_fea_out, mixed_y, args.temperature, args.lamda)
            init_loss += loss.item() * y.size(0)
            init_acc += (adv_out_soft.argmax(1) == y).sum().item()

            loss.backward(retain_graph=True)
            grad = F.interpolate(delta.grad.detach(), size=(64, 64), mode='bilinear', align_corners=False)
            delta.data = clamp(delta + alpha_pixel * grad.sign(), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - mixed_X, upper_limit - mixed_X)
            delta = delta.detach()
            ori_out, fea_out = model(mixed_X + delta)
            ori_out_soft = F.softmax(ori_out, dim=1)
            fea_out_soft = F.softmax(fea_out, dim=1)
            mse = nn.MSELoss(reduction='mean') 
            lbl_sm = _label_smoothing(y, args.factor)

            loss_final = label_smooth_loss(ori_out, lbl_sm) + args.lamda * (
                mse(ori_out_soft, adv_out_soft) + mse(fea_out_soft, ori_fea_out_soft)
            ) / (mse(mixed_X + delta, mixed_X + temp_delta) + 0.125)

            optim.zero_grad()
            loss_final.backward()
            optim.step()

            running_loss += loss_final.item() * y.size(0)
            running_acc += (ori_out.argmax(1) == y).sum().item()
            running_n += y.size(0)

            adv_acc = (ori_out_soft.argmax(1) == mixed_y.argmax(1)).sum().item()
            clean_acc = (adv_out_soft.argmax(1) == mixed_y.argmax(1)).sum().item()
            if adv_acc / (clean_acc + 1e-8) < args.ema_value:
                teacher.update(model)
                teacher.apply_shadow()

            scheduler.step()
        epoch_time = time.time() - epoch_start
        train_loss = running_loss / running_n
        train_acc = running_acc / running_n
        lr_now = scheduler.get_last_lr()[0]
        logger.info(f"{epoch:03d}\t{epoch_time:.1f}s\t{lr_now:.4f}\t{train_loss:.4f}\t{train_acc:.4f}")
        model_test = copy.deepcopy(model).to(device)
        model_test.load_state_dict(teacher.model.state_dict(), strict=True)
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(val_loader, model_test, 10, 1)
        val_loss, val_acc = evaluate_standard(val_loader, model_test)
        clean_acc_trend.append(val_acc)
        pgd_acc_trend.append(pgd_acc)
        logger.info(f"ValLoss {val_loss:.4f}\tValAcc {val_acc:.4f}\tPGDLoss {pgd_loss:.4f}\tPGDAcc {pgd_acc:.4f}")

        if pgd_acc >= best_pgd_acc:
            best_pgd_acc = pgd_acc
            torch.save(model_test.state_dict(), os.path.join(out_dir, 'best_model.pth'))
        
    torch.save(model_test.state_dict(), os.path.join(out_dir, 'final_model.pth'))
    logger.info('Training complete')
    logger.info(clean_acc_trend)
    logger.info(pgd_acc_trend)
    print(clean_acc_trend)
    print(pgd_acc_trend)

if __name__ == '__main__':
    main()
