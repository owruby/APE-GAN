# APE-GAN

Implementation APE-GAN (https://arxiv.org/pdf/1707.05474.pdf)

## MNIST
### 1. Generate CNN and Adversarial Examples(FGSM)
```
python generate.py --eps 0.15
```

### 2. Train APE-GAN
```
python train.py --checkpoint ./checkpoint/mnist
```

### 3. Test
```
python test_model.py --eps 0.15 --gan_path ./checkpoint/mnist/3.tar
```
