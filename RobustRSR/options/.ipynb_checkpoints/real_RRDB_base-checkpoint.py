# coding: utf-8
import os
"""
SRResNet 在real pair上实验，使用真实图像对
"""

'''
# Train Data Root
’/mnt/lustre/luhannan/Data_t1/PairHL_crop800_6a/‘
'/mnt/lustre/luhannan/Data_t1/PairHL_crop800_5c/'
'/mnt/lustrenew/luhannan/dataset/PairHL_crop800_7b' # X2

# Test Data Root
'./testsets/test_c400_x2/'
'./testsets/test_HL_c200/'
'./testsets/real_vali_c200_20/'
'''

import argparse

def parse_config(local_test=True):
    parser = argparse.ArgumentParser()
    ## For Local Test
    if local_test:
        parser.add_argument('--batch_size', type=int, default=2, help='The max input batch size')
        parser.add_argument('--size', type=int, default=60, help='the low resolution image size')
        parser.add_argument('--dataroot', type=str, default='testsets/test_LR_c500', help='path to dataset')
        parser.add_argument('--test_dataroot', type=str, default='testsets/Set5', help='path to dataset')
        parser.add_argument('--test_interval', type=int, default=1, help="path to generator weights A(to continue training)")
        parser.add_argument('--pretrain', type=str, default='', help='folder to output model checkpoints')
        parser.add_argument('--vgg', type=str, default='', help='number of threads to prepare data.')
        parser.add_argument('--use_cuda', type=bool, default=False, help='number of threads to prepare data.')
        # parser.add_argument('--use_cuda', type=bool, default=True, help='number of threads to prepare data.')
        parser.add_argument('--bic', type=bool, default=False, help='[normal | real_pair]')
    ## For Server
    else:
        # parser.add_argument('--dataroot', type=str, default='/mnt/lustre/luhannan/Data_t1/photo_process/5_camera_4_warp2/train_patch_done', help='Train dataset')
        parser.add_argument('--dataroot', type=str, default='/mnt/lustre/luhannan/Data_t1/NTIRE_19/ntire_train_55', help='Train dataset')
        # parser.add_argument('--dataroot', type=str, default='/mnt/lustre/luhannan/Data_t1/PairHL_crop800_6a/', help='Train dataset')
        # parser.add_argument('--dataroot', type=str, default='/mnt/lustre/luhannan/Data_t1/PairHL_crop800_5c/', help='Train dataset')
        # parser.add_argument('--dataroot', type=str, default='/mnt/lustre/luhannan/Data_t1/ntire_train_1/', help='Train dataset')
        # parser.add_argument('--dataroot', type=str, default='/mnt/lustrenew/luhannan/dataset/PairHL_crop800_5c', help='Train dataset')
        # parser.add_argument('--dataroot', type=str, default='/mnt/lustrenew/luhannan/dataset/PairHL_crop800_6a', help='Train dataset')
        # parser.add_argument('--test_dataroot', type=str, default='./testsets/real_vali_c200_20/', help='Test dataset')
        # parser.add_argument('--test_dataroot', type=str, default='./testsets/ntire_vali_1/', help='Test dataset')
        # parser.add_argument('--test_dataroot', type=str, default='./testsets/real_vali_c200_24/', help='Test dataset')
        parser.add_argument('--test_dataroot', type=str, default='/mnt/lustre/luhannan/Data_t1/NTIRE_19/ntire_train_55/valid_patch', help='Test dataset')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
        parser.add_argument('--size', type=int, default=200, help='Size of low resolution image')
        parser.add_argument('--test_interval', type=int, default=1, help="Test epoch")
        parser.add_argument('--pretrain', type=str, default='./experiments/ntire_l1_p48/checkpoint/SRResNet_X1_800.pth', help='Pretrained SRResNet weight')
        # parser.add_argument('--pretrain', type=str, default='./experiments/ntire_l1_p196/checkpoint/SRResNet_X1_200.pth', help='Pretrained SRResNet weight')
        # parser.add_argument('--pretrain', type=str, default='./models/RRDB_PSNR_x4_0.pth', help='Pretrained SRResNet weight')
        parser.add_argument('--vgg', type=str, default='./models/vgg19.pth', help='Pretrained VGG weight')
        parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use GPU')
        parser.add_argument('--bic', type=bool, default=False, help='Use bicubic downsample or not, False for real pair')

    ## Experiment Settings
    parser.add_argument('--exp_name', type=str, default='ntire_l1_p200_emask', help='folder to output model checkpoints')
    parser.add_argument('--exp_info', type=str, default='用edge mask从p48-800开始训练，p200', help='')
    parser.add_argument('--no_HR', type=bool, default=False, help='Whether these are HR images in testset or not')
    parser.add_argument('--use_edge_mask', type=bool, default=True, help='Whether these are HR images in testset or not')
    parser.add_argument('--best_psnr', type=float, default=29.45, help='Whether these are HR images in testset or not')

    ## Train Settings
    parser.add_argument('--generatorLR', type=float, default=2e-4, help='learning rate for SR generator')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of threads to prepare data.')
    parser.add_argument('--lr_decay', type=list, default=[2e5, 4e5, 6e5], help='number of threads to prepare data.')

    ## NetG Settings
    parser.add_argument('--model', type=str, default='RRDB_net', help='[RRDB_net | srresnet]')
    parser.add_argument('--sr_norm_type', default=None, help='For SRResNet, IN or No IN [None | IN ]')
    parser.add_argument('--scala', type=int, default=1, help='[1 | 2 | 4], 1 for NTIRE Challenge')
    parser.add_argument('--in_ch', type=int, default=3, help='Image channel, 3 for RGB')
    parser.add_argument('--out_ch', type=int, default=3, help='Image channel, 3 for RGB')
    parser.add_argument('--rrdb_nf', type=int, default=64, help='For RRDB, Feature Number for Conv')
    parser.add_argument('--rrdb_nb', type=int, default=23, help='For RRDB, Blocks Number for RRD-block')
    parser.add_argument('--rrdb_gc', type=int, default=32, help='For RRDB, Blocks Number for RRD-block')
    parser.add_argument('--rrdb_group', type=int, default=1, help='For RRDB, Blocks Number for RRD-block')

    ## Contextual Loss Settings
    parser.add_argument('--cx_loss', type=bool, default=False, help='')
    parser.add_argument('--cx_loss_lambda', type=float, default=1e-1, help='Weight for CX_Loss')
    parser.add_argument('--cx_vgg_layer', type=int, default=34, help='[17 | 34]')

    # Content Loss Settings
    parser.add_argument('--sr_lambda', type=float, default=1, help='content loss lambda')
    parser.add_argument('--loss', type=str, default='l1', help="content loss ['l1', 'l2', 'c']")

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

    ## STN Settings
    parser.add_argument('--use_stn', type=bool, default=False, help='Whether downsample test LR image')
    parser.add_argument('--angle', type=int, default=60)
    parser.add_argument('--span_range', type=int, default=0.9)
    parser.add_argument('--grid_size', type=int, default=4)
    parser.add_argument('--stn_model', type=str, default='unbounded_stn')
    parser.add_argument('--stn_lr', type=float, default=1e-2)
    parser.add_argument('--stn_momentum', type=float, default=0.5)
    parser.add_argument('--sr_label', type=bool, default=False, help='Use SR Image to warp HR')

    ## Previous Exp Settings
    parser.add_argument('--sub_mean', type=bool, default=False, help='Whether use mean to correct HR`s brightness[no use]')
    parser.add_argument('--shift', type=bool, default=False, help='Whether random transform input image')

    ## Default Settings
    parser.add_argument('--gpus', type=int, default=1, help='Placeholder, will be changed in run_train.sh')
    parser.add_argument('--train_file', type=str, default='', help='placeholder, will be changed in main.py')
    parser.add_argument('--config_file', type=str, default='', help='placeholder, will be changed in main.py')

    opt = parser.parse_args()

    return opt

























