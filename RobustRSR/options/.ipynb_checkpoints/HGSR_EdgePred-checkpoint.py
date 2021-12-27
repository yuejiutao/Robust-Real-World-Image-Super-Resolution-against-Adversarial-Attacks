# coding: utf-8
import os
"""
Config File for HGSR-Multi-Scale-Supervision
"""

import argparse

def parse_config(local_test=True):
    parser = argparse.ArgumentParser()

    # Data Preparation
    parser.add_argument('--dataroot', type=str, default='/home/weipx/RealSR/Dataset/5cWarp3_1')
    parser.add_argument('--test_dataroot', type=str, default='/home/weipx/RealSR/Dataset/5cWarp3_1/Vali100_rand36')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--size', type=int, default=48, help='Size of low resolution image')
    parser.add_argument('--bic', type=bool, default=False, help='')
    parser.add_argument('--rgb_range', type=float, default=1., help='255 EDSR and RCAN, 1 for the rest')
    parser.add_argument('--no_HR', type=bool, default=False, help='Whether these are HR images in testset or not')

    ## Train Settings
    parser.add_argument('--exp_name', type=str, default='HGSRn6Ori_EdgePred_BCE', help='')
    parser.add_argument('--generatorLR', type=float, default=1e-4, help='learning rate for SR generator')
    parser.add_argument('--decay_step', type=list, default=[1e4, 2e4, 3e4, 8e5], help='')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--test_interval', type=int, default=1, help="Test epoch")
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use GPU')
    parser.add_argument('--workers', type=int, default=8, help='number of threads to prepare data.')

    parser.add_argument('--skip_threshold', type=int, default=1e4, help='number of feature maps')

    ## SRModel Settings
#     parser.add_argument('--pretrain', type=str, default='./experiments/HGSRn6_ori_EdgePred/checkpoint/HGSR_X4_117.pth')
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--model', type=str, default='HGSR', help='[RRDB | srresnet | EDSR | RCAN | HGSR | HGSR-MHR]')
    parser.add_argument('--scala', type=int, default=4, help='[1 | 2 | 4], 1 for NTIRE Challenge')
    parser.add_argument('--in_ch', type=int, default=4, help='Image channel, 3 for RGB')
    parser.add_argument('--out_ch', type=int, default=1, help='Image channel, 3 for RGB')

    # HGSR Settings
    parser.add_argument('--n_HG', type=int, default=6, help='number of feature maps')
    parser.add_argument('--res_type', type=str, default='res', help='residual scaling')
    parser.add_argument('--inter_supervis', type=bool, default=False, help='residual scaling')
    parser.add_argument('--inte_loss_weight', type=list, default=[1e-1, 5e-1, 1], help='residual scaling')
    
    parser.add_argument('--rrdb_nf', type=int, default=64, help='For RRDB, Feature Number for Conv')
    parser.add_argument('--rrdb_nb', type=int, default=23, help='For RRDB, Blocks Number for RRD-block')
    parser.add_argument('--rrdb_gc', type=int, default=32, help='For RRDB, Blocks Number for RRD-block')
    parser.add_argument('--rrdb_group', type=int, default=1, help='For RRDB, Blocks Number for RRD-block')

    # Content Loss Settings
    parser.add_argument('--sr_lambda', type=float, default=1, help='content loss lambda')
    parser.add_argument('--loss', type=str, default='l1', help="content loss ['l1', 'l2', 'c']")

    ## Prior Loss Settings
    parser.add_argument('--prior_loss', type=bool, default=False, help='')
    parser.add_argument('--prior_lambda', type=float, default=1, help='')
    parser.add_argument('--downsample_kernel', type=str, default='lanczos2', help='')

    ## Contextual Loss Settings
    parser.add_argument('--cx_loss', type=bool, default=False, help='')
    parser.add_argument('--cx_loss_lambda', type=float, default=1e-1, help='Weight for CX_Loss')
    parser.add_argument('--cx_vgg_layer', type=int, default=34, help='[17 | 34]')

    ## VGG Loss Settings
    parser.add_argument('--vgg_loss', type=bool, default=False, help='Whether downsample test LR image')
    parser.add_argument('--vgg_lambda', type=float, default=1, help='learning rate for generator')
    parser.add_argument('--vgg_loss_type', type=str, default='l1', help="loss L1 or L2 ['l1', 'l2']")
    parser.add_argument('--vgg_layer', type=str, default=34, help='[34 | 35]')

    ## TV Loss Settings
    parser.add_argument('--tv_loss', type=bool, default=False, help='Whether use tv loss')
    parser.add_argument('--tv_lambda', type=float, default=0.1, help='tv loss lambda')

    ## Noise Settings
    parser.add_argument('--add_noise', type=bool, default=False, help='Whether downsample test LR image')
    parser.add_argument('--noise_level', type=float, default=0.1, help='learning rate for SR generator')

    ## Mix Settings
    parser.add_argument('--mix_bic_real', type=bool, default=False, help='Whether mix ')

    ## Default Settings
    parser.add_argument('--gpus', type=int, default=1, help='Placeholder, will be changed in run_train.sh')
    parser.add_argument('--train_file', type=str, default='', help='placeholder, will be changed in main.py')
    parser.add_argument('--config_file', type=str, default='', help='placeholder, will be changed in main.py')

    opt = parser.parse_args()

    return opt

























