<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Enhancing Fast Adversarial Training Efficiency with Mixup Augmentation and Geometric Temperature-Scaled KL Divergence Loss </h1>
<p align='left' style="text-align:left;font-size:1.2em;">
</p>

## Introduction


> This work investigates methods for improving the efficiency of fast adversarial training through the integration of Mixup data augmentation and Geometric temperature-scaled KL divergence loss. Catastrophic overfitting often occurs during adversarial training, where models become overly specialized on adversarial examples, leading to poor generalization on clean data. To address this challenge, the Mixup technique enhances the smoothness of decision boundaries, thereby facilitating improved generalization in learning processes. The Geometric Temperature-Scaled KL Divergence Loss facilitates teacher-student distillation by dynamically transitioning the learning process from soft "dark knowledge" to high-confidence predictions, thereby ensuring stability during training. The combination of these elements reduces the risk of severe overfitting, enhances generalization capabilities, and promotes more consistent training dynamics. 
Experimental results show that combining Mixup and Geometric temperature-scaled KL divergence loss significantly improves adversarial accuracy, reduces overfitting, and accelerates the training process, outperforming traditional adversarial training methods. Our work provides a more efficient and stable framework for adversarial training in deep learning models.



## Train
```
python3 MIX_KL_CIFAR10.py  --out_dir ./output/ --data-dir cifar10-data
python3 MIX_KL_CIFAR100.py  --out_dir ./output/ --data-dir cifar100-data

```

## Test
```
python3.6 test_cifar10.py --model_path model.pth --out_dir ./output/ --data-dir cifar10-data
python3.6 test_cifar100.py --model_path model.pth --out_dir ./output/ --data-dir cifar100-data

```

## Trained Models
> The Trained models will be available soon
