
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from CustomDataset import CustomDataset

from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os  # lv

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        # self.loss_fn = nn.MSELoss()

    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        degrad_patch, gt_patch = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, gt_patch)

        self.log("train_loss", loss, prog_bar=True)  # 多卡训练指定sync_dist=True
        return loss
    
    def validation_step(self, batch, batch_idx):
        degrad_patch, gt_patch = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, gt_patch)

        self.log("val_loss", loss, prog_bar=True)  
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()
    
    def configure_optimizers(self): 
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=2e-4)  # 根据requires_grad绑定参数
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=opt.epochs // 10, max_epochs=opt.epochs) 

        return [optimizer],[scheduler]


def main():

    """指定数据集路径以及加载的权重"""
    train_degradation_path = r''
    train_gt_path = r''

    val_degradation_path = r''
    val_gt_path = r''

    pretrained_path = r'./ckpt/model.ckpt'

    # ############ logger ################
    
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = CustomDataset(degradation_path=train_degradation_path, gt_path=train_gt_path, train=True)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=opt.num_workers)

    valset = CustomDataset(degradation_path=val_degradation_path, gt_path=val_gt_path, train=False)
    valloader = DataLoader(valset, batch_size=opt.batch_size, pin_memory=True, shuffle=False, drop_last=False, num_workers=opt.num_workers)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath = opt.ckpt_dir,
        # every_n_epochs = 10,
        filename='best_ckpt',
        save_top_k=1
    )
    
    # 加载全部权重
    model = PromptIRModel.load_from_checkpoint(pretrained_path)

    # ##################### 冻结bacbone###################

    for param in model.net.parameters():
        param.requires_grad = False

    for param in model.net.prompt1.parameters():
        param.requires_grad = True
    for param in model.net.noise_level1.parameters():
        param.requires_grad = True
    for param in model.net.reduce_noise_level1.parameters():
        param.requires_grad=True

    for param in model.net.prompt2.parameters():
        param.requires_grad = True
    for param in model.net.noise_level2.parameters():
        param.requires_grad = True
    for param in model.net.reduce_noise_level2.parameters():
        param.requires_grad=True


    for param in model.net.prompt3.parameters():
        param.requires_grad = True
    for param in model.net.noise_level3.parameters():
        param.requires_grad = True
    for param in model.net.reduce_noise_level3.parameters():
        param.requires_grad=True
    
    # ################统计参数量#########################

    # prompt1_params = sum(p.numel() for p in model.net.prompt1.parameters())
    # prompt2_params = sum(p.numel() for p in model.net.prompt2.parameters())
    # prompt3_params = sum(p.numel() for p in model.net.prompt3.parameters())

    # noise_l1_params = sum(p.numel() for p in model.net.noise_level1.parameters())
    # noise_l2_params = sum(p.numel() for p in model.net.noise_level2.parameters())
    # noise_l3_params = sum(p.numel() for p in model.net.noise_level3.parameters())

    # reduce_noise_l1_params = sum(p.numel() for p in model.net.reduce_noise_level1.parameters())
    # reduce_noise_l2_params = sum(p.numel() for p in model.net.reduce_noise_level2.parameters())
    # reduce_noise_l3_params = sum(p.numel() for p in model.net.reduce_noise_level3.parameters())

    # latent_params = sum(p.numel() for p in model.net.latent.parameters())  # 这个是超级大头，一个就占了40%
    
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    # print(f'reduce_noise_l1_params: {reduce_noise_l1_params}')
    # print(f'reduce_noise_l2_params: {reduce_noise_l2_params}')
    # print(f'reduce_noise_l3_params: {reduce_noise_l3_params}')

    # print(f'latent_params: {latent_params}')

    # print(f'total_params: {total_params / 1024 / 1024:.2f} M')
    # print(f'trainable_params: {trainable_params / 1024 / 1024:.2f} M')

    # ####################################################

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,

        # strategy="ddp",
        precision=16,
        logger=logger,
        callbacks=checkpoint_callback
        # logger=None,
        # callbacks=None
    )

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)

if __name__ == '__main__':
    main()