<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Enhancing Fast Adversarial Training Efficiency with Mixup Augmentation and Temperature-Scaled KL Divergence Loss </h1>
<p align='left' style="text-align:left;font-size:1.2em;">
</p>

## Introduction
<p align="center">
Generation process of adversarial examples in Fast Adversarial Training (FAT) using Mixup Augmentation and Temperature-Scaled KL Divergence Loss: (a) Using Mixup Augmentation:. (b) Using Temperature-Scaled KL Divergence Loss: (c)Combining Mixup and Temperature Scaling:
</p>


> This work investigates methods for improving the efficiency of fast adversarial training through the integration of Mixup data augmentation and temperature-scaled KL divergence loss. Catastrophic overfitting often occurs during adversarial training, where models become overly specialized on adversarial examples, leading to poor generalization on clean data. To address this challenge, we propose incorporating Mixup augmentation, which synthesizes new training examples by interpolating between pairs of samples, to enhance generalization and reducing the risk of overfitting. Additionally, we introduce temperature-scaled KL divergence loss as a means to optimize the adversarial training process by adjusting the distribution of predictions and reducing the impact of noisy or outlier gradients. Our approach aims to enhance both adversarial robustness and training efficiency while preventing catastrophic overfitting. Experimental results show that combining Mixup and temperature-scaled KL divergence loss significantly improves adversarial accuracy, reduces overfitting, and accelerates the training process, outperforming traditional adversarial training methods. Our work provides a more efficient and stable framework for adversarial training in deep learning models.



## Train
```
python3 MIX_KL_CIFAR10.py  --out_dir ./output/ --data-dir cifar-data
python3 MIX_KL_CIFAR100.py  --out_dir ./output/ --data-dir cifar-data

```

## Test
```
python3.6 test_cifar10.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data
python3.6 test_cifar100.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data

```

## Trained Models
> The Trained models will be available soon