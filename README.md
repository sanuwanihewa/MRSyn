# MRI Synthesis using Adversarial Diffusion

![](https://github.com/sanuwanihewa/MRSyn/blob/main/figures/animated.gif)

### Dataset Preparation

We will be using 100 middle axial slices of T2 and FLAIR contrasts from Brats dataset and save each as .npy file.
You can use the python script to prepare the dataset or download sample data from

The structure of the dataset should be as follows.
```
data/
├── BRATS/
│   ├── train/
│   │   ├── T2.npy
│   │   └── Flair.npy
│   ├── test/
│   │   ├── T2.npy
│   │   └── Flair.npy
│   ├── val/
│   │   ├── T2.npy
│   │   └── Flair.npy
```

### Model Training

``` 
python train.py --image_size 256 --exp exp_syn--num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4  --num_epoch 50 --ngf 64 --embedding_type positional --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 1
```

### Model Testing

``` 
python test.py --image_size 256 --exp exp_syn --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional  --z_emb_dim 256  --which_epoch 50 --gpu_chose 0 --input_path '/data/shew0029/MedSyn/DATA/BRATS/' --output_path '/data/shew0029/MedSyn/VQGAN_3D/mrisyn/results'
```
