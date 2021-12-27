import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.functional import pad as tensor_pad
from torchvision import transforms
import numpy as np
import cv2
import pdb
import matplotlib
matplotlib.use('Agg')

import time
import argparse
import sys
sys.path.append('../')
sys.path.append('./modules')
sys.path.append('./modules/PerceptualSimilarity')
# sys.path.append('./modules/pytorch_ssim')
import PIL
import PIL.Image
from io import BytesIO
import numpy as np

import os
from TorchTools.DataTools.DataSets import SRDataList, RealPairDataset,RealPairDatasetTest
from TorchTools.Functions.Metrics import PSNR, psnr, YCbCr_psnr
from TorchTools.TorchNet.tools import calculate_parameters, load_weights
from TorchTools.Functions.functional import tensor_block_cat, tensor_block_crop, tensor_merge, tensor_divide
from TorchTools.DataTools.Loaders import to_pil_image, to_tensor
import pytorch_ssim
import random
import warnings


os.environ["CUDA_VISIBLE_DEVICES"] = "6"
warnings.filterwarnings("ignore")

local_test = False
debug_mode = False
use_cuda = True

parser = argparse.ArgumentParser()
# Test Dataset
parser.add_argument('--test_dataroot', type=str, default='')
parser.add_argument('--mul', type=int, default=100)
parser.add_argument('--model', type=str, default='SRENCNET',help='[srresnet | RRDB_net | EDSR]')
parser.add_argument('--pretrain', type=str, default='')
parser.add_argument('--ID', type=int, default=1)
parser.add_argument('--testdown', type=int, default=1)
parser.add_argument('--testup', type=int, default=1)
parser.add_argument('--n_HG', type=int, default=6, help='the low resolution image size')
parser.add_argument('--advtrain', type=bool, default=False)
parser.add_argument('--save_results', type=bool, default=False, help='Concat result to one image')

# Generate adversarial sample
parser.add_argument('--nb_iter',type=int,default=10)


# Test Options
parser.add_argument('--res_type', type=str, default='res')
parser.add_argument('--overlap', type=int, default=32, help='Overlap pixel when Divide input image, for edge effect')
parser.add_argument('--psize', type=int, default=224, help='Overlap pixel when Divide input image, for edge effect')
parser.add_argument('--real', type=bool, default=True, help='Whether to downsample input image')
parser.add_argument('--cat_result', type=bool, default=False, help='Concat result to one image')
parser.add_argument('--has_GT', type=bool, default=True, help='the low resolution image size')
parser.add_argument('--rgb_range', type=float, default=1., help='255 EDSR and RCAN, 1 for the rest')
parser.add_argument('--bic', type=bool, default=False, help='Concat result to one image')

# Model Options
parser.add_argument('--sr_norm_type', default='IN', help='[srresnet | RRDB_net]')
parser.add_argument('--rrdb_nb', type=int, default=23, help='For RRDB, Blocks Number for RRD-block')
parser.add_argument('--inc', type=int, default=3, help='the low resolution image size')
parser.add_argument('--scala', type=int, default=4, help='the low resolution image size')

parser.add_argument('--inter_supervis', type=bool, default=True, help='the low resolution image size')
parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
parser.add_argument('--n_colors', type=int, default=3, help='number of feature maps')
parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test (single | half)')

parser.add_argument('--seed', type=int, default=123)

# Barely change
parser.add_argument('--workers', type=int, default=8, help='number of threads to prepare data.')
# Logger
parser.add_argument('--result_dir', type=str, default='hgsr_x4_test', help='folder to sr results')
parser.add_argument('--gpus', type=int, default=1, help='folder to sr results')
opt = parser.parse_args()


######## other functions ####################################################################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Init Net SR Model
def get_model(opt):
    print('...Build Generator Net...')
    if (opt.model == 'CDC_MC'):
        sys.path.append('./modules/HGSR')
        import model
        generator = model.CDC_MC(in_nc=opt.inc, out_nc=opt.inc, upscale=opt.scala,
                                     nf=64, res_type=opt.res_type, n_mid=2,
                                     n_HG=opt.n_HG, inter_supervis=opt.inter_supervis)
    return generator
#############################################################################################




######## Generate SR images under 0 ##########################################################
setup_seed(opt.seed)
device = torch.device('cuda') if use_cuda else torch.device('cpu')

# Make Result Folder
opt.result_dir = os.path.abspath(os.path.join( os.getcwd(), "../../../Data/RealSR/x4/adv/%s" % opt.model))
if not os.path.exists(opt.result_dir):
    os.makedirs(opt.result_dir)

# Save Test Info
log_f = open(os.path.join(opt.result_dir, 'test_log.txt'), 'a')
log_f.write('test_dataroot: ' + opt.test_dataroot + '\n')
log_f.write('real: ' + str(opt.real) + '\n')
log_f.write('test_patch_size: ' + str(opt.psize) + '\n')
log_f.write('test_overlap: ' + str(opt.overlap) + '\n')
log_f.write('test_model: ' + str(opt.pretrain) + '\n')
log_f.write('result_dir: ' + str(opt.result_dir) + '\n')

# Init Dataset
if opt.has_GT and (not opt.bic):
    test_dataset = RealPairDatasetTest(os.path.join(opt.test_dataroot, 'test_LR'), os.path.join(opt.test_dataroot, 'test_HR'), lr_patch_size=0, scala=opt.scala,
                                       mode='Y' if opt.inc == 1 else 'RGB', train=False, need_hr_down=False,
                                       need_name=True, rgb_range=opt.rgb_range)
else:
    test_dataset = SRDataList(opt.test_dataroot, lr_patch_size=0, scala=opt.scala, mode='Y' if opt.inc == 1 else 'RGB',
                              train=False, need_name=True, rgb_range=opt.rgb_range)
test_loader = DataLoader(test_dataset, batch_size=1,
                         shuffle=False, num_workers=opt.workers, drop_last=True)

print("*** opt.model ", opt.model)
print("*** opt.rgb_range ", opt.rgb_range)
print("*** opt.save_results ", opt.save_results)

generator = get_model(opt)

# load weights
pretrain = opt.pretrain
print("*** pretrain ", pretrain)
generator = load_weights(generator, pretrain, opt.gpus, just_weight=False, strict=True)
generator = generator.to(device)
generator.eval()

psnr_sum = 0
ssim_sum = 0

for batch, data in enumerate(test_loader):
    lr = data['LR']
    hr = data['HR']
    im_path = data['HR_PATH'][0]

    hr_var = hr.to(device)

    with torch.no_grad():
        if opt.has_GT:
            tensor = lr
            hr_size = hr
        else:
            tensor = hr
            B, C, H, W = hr.shape
            hr_size = torch.zeros(B, C, H * opt.scala, W * opt.scala)
        blocks = tensor_divide(tensor, opt.psize, opt.overlap)
        blocks = torch.cat(blocks, dim=0)
        results = []

        iters = blocks.shape[0] // opt.gpus if blocks.shape[0] % opt.gpus == 0 else blocks.shape[0] // opt.gpus + 1
        for idx in range(iters):
            if idx + 1 == iters:
                input = blocks[idx * opt.gpus:]
            else:
                input = blocks[idx * opt.gpus: (idx + 1) * opt.gpus]
            lr_var = input.to(device)

            if (opt.model == 'CDC_MC'):
                sr_var, _, _, Flag = generator(lr_var)
                sr_var = sr_var[-1]

            results.append(sr_var)
            print('Processing Image: %d Part: %d / %d' %
                  (batch + 1, idx + 1, iters), end='\r')
            sys.stdout.flush()

        results = torch.cat(results, dim=0)
        sr_img = tensor_merge(results, hr_var, opt.psize *
                              opt.scala, opt.overlap * opt.scala)

    if opt.has_GT:
        psnr_single = YCbCr_psnr(sr_img.to(device), hr_var, scale=opt.scala, peak=opt.rgb_range)
        with torch.no_grad():
            ssim_single = pytorch_ssim.ssim(sr_img.to(device) / opt.rgb_range, hr.to(device) / opt.rgb_range).item()
        psnr_sum += psnr_single
        ssim_sum += ssim_single
    sys.stdout.flush()

    ##### save #####
    im_name = '%s_%s_x%d.png' % (os.path.basename(im_path).split('.')[0], opt.model, opt.scala)
    img_save_dir = os.path.join(opt.result_dir, '0/test_SR')

    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    if opt.save_results:
        im = to_pil_image(torch.clamp(sr_img[0].cpu() / opt.rgb_range, min=0.0, max=1.0))
        im.save(os.path.join(img_save_dir, im_name))
        print('[%d/%d] saving to: %s' %(batch + 1, len(test_loader), os.path.join(img_save_dir, im_name)))

if opt.has_GT:
    psnr_sum /= len(test_loader)
    ssim_sum /= len(test_loader)
    print('-----------\nAve: PSNR: %.4f SSIM: %.4f\n-----------' %(psnr_sum, ssim_sum))
    log_f.write('-----------\nAve: PSNR: %.4f SSIM: %.4f\n-----------' %(psnr_sum, ssim_sum))
    sys.stdout.flush()
















######## Generate adversarial samples ###################################################
IFGSM_FLAG=True
setup_seed(456)

# Init Dataset
if opt.has_GT and (not opt.bic):
    test_dataset = RealPairDatasetTest(os.path.join(opt.test_dataroot,'test_LR'), os.path.join(opt.result_dir,'0/test_SR'), lr_patch_size=0, scala=opt.scala,
                                   mode='Y' if opt.inc == 1 else 'RGB', train=False, need_hr_down=False,
                                   need_name=True, rgb_range=opt.rgb_range)
else:
    test_dataset = SRDataList(opt.test_dataroot, lr_patch_size=0, scala=opt.scala, mode='Y' if opt.inc == 1 else 'RGB',
                          train=False, need_name=True, rgb_range=opt.rgb_range)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers, drop_last=True)

# Init ifgsm algorithm
if IFGSM_FLAG:
    sys.path.append('./modules/IFGSM')
    sys.path.append('./modules/IFGSM/model')
    import ifgsm
    ifgsm_model=ifgsm.make_model(srmodel=generator,modelname=opt.model)

adv_rate=[1,2,4,6,8]
print("*** opt.nb_iter ",opt.nb_iter)

for i in range(len(adv_rate)):
    rate=adv_rate[i]*1.0

    if 'EDSR' in opt.model or 'RCAN' in opt.model:
        nb_iter=opt.nb_iter; clip_min=0.0; clip_max=255.0; clip_eps_min=-rate; clip_eps_max=rate;
    else :
        rate= rate/255.0; nb_iter=opt.nb_iter; clip_min=0.0; clip_max=1.0; clip_eps_min=-rate; clip_eps_max=rate;

    for batch, data in enumerate(test_loader):
        lr = data['LR']
        hr = data['HR']
        im_path = data['LR_PATH'][0]
        im_name=im_path.split('/')[-1]

        if opt.has_GT:
            tensor = lr
            hr_size = hr
        else:
            tensor = hr
            B, C, H, W = hr.shape
            hr_size = torch.zeros(B, C, H * opt.scala, W * opt.scala)

        blocks = tensor_divide(tensor, opt.psize, opt.overlap)
        blocks = torch.cat(blocks, dim=0)

        blocks2 = tensor_divide(hr_size, opt.psize * opt.scala, opt.overlap * opt.scala)
        blocks2 = torch.cat(blocks2, dim=0)
        results = []

        iters = blocks.shape[0] // opt.gpus if blocks.shape[0] % opt.gpus == 0 else blocks.shape[0] // opt.gpus + 1
        for idx in range(iters):
            if idx + 1 == iters:
                input = blocks[idx * opt.gpus:]
                input2 = blocks2[idx * opt.gpus:]
            else:
                input = blocks[idx * opt.gpus: (idx + 1) * opt.gpus]
                input2 = blocks2[idx * opt.gpus: (idx + 1) * opt.gpus]
            lr_var = input.to(device)
            gt_var = input2.to(device)

            adv_lr = ifgsm_model.generate(lr_var, gt_var, scala=4, eps=rate, iter_eps=rate, nb_iter=nb_iter,
                                          clip_min=clip_min,
                                          clip_max=clip_max, clip_eps_min=clip_eps_min,
                                          clip_eps_max=clip_eps_max,
                                          rand_init=False)

            results.append(adv_lr.to('cpu'))
            print('Processing Image: %d Part: %d / %d' % (batch + 1, idx + 1, iters), end='\r')
            sys.stdout.flush()

        results = torch.cat(results, dim=0)
        sr_img = tensor_merge(results, tensor, opt.psize, opt.overlap)

        im_name = 'adv' + str(adv_rate[i]) + im_name
        img_save_dir = os.path.join(opt.result_dir,str(adv_rate[i])+'/test_LR')
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        if opt.save_results:
            im = to_pil_image(torch.clamp(sr_img[0].cpu() / opt.rgb_range, min=0.0, max=1.0))
            im.save(os.path.join(img_save_dir, im_name))
            print('[%d/%d] saving to: %s' % (batch + 1, len(test_loader), os.path.join(img_save_dir, im_name)))




######## Generate SR images for adversarial samples ##########################################
setup_seed(789)
adv_rate = [8,6,4,2,1]
psnr_list = []
ssim_list = []

for ii in range(len(adv_rate)):
    rate = adv_rate[ii]

    # Init Dataset
    if opt.has_GT and (not opt.bic):
        test_dataset = RealPairDatasetTest(os.path.join(opt.result_dir,'%d/test_LR'%rate ), os.path.join(opt.test_dataroot,'test_HR'), lr_patch_size=0, scala=opt.scala,
                                           mode='Y' if opt.inc == 1 else 'RGB', train=False, need_hr_down=False,
                                           need_name=True, rgb_range=opt.rgb_range)
    else:
        test_dataset = SRDataList(opt.test_dataroot, lr_patch_size=0, scala=opt.scala,
                                  mode='Y' if opt.inc == 1 else 'RGB',
                                  train=False, need_name=True, rgb_range=opt.rgb_range)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers, drop_last=True)

    psnr_sum = 0
    ssim_sum = 0

    for batch, data in enumerate(test_loader):
        lr = data['LR']
        hr = data['HR']
        im_path = data['LR_PATH'][0]
        im_name = im_path.split('/')[-1]

        with torch.no_grad():
            if opt.has_GT:
                tensor = lr
                hr_size = hr
            else:
                tensor = hr
                B, C, H, W = hr.shape
                hr_size = torch.zeros(B, C, H * opt.scala, W * opt.scala)

            blocks = tensor_divide(tensor, opt.psize, opt.overlap)
            blocks = torch.cat(blocks, dim=0)
            results = []

            iters = blocks.shape[0] // opt.gpus if blocks.shape[0] % opt.gpus == 0 else blocks.shape[0] // opt.gpus + 1
            for idx in range(iters):
                if idx + 1 == iters:
                    input = blocks[idx * opt.gpus:]
                else:
                    input = blocks[idx * opt.gpus: (idx + 1) * opt.gpus]
                lr_var = input.to(device)

                if opt.model=='CDC_MC':
                    sr_var, _, _, Flag = generator(lr_var)
                    # print("*** Flag ",Flag)
                    sr_var = sr_var[-1]
                else:
                    sr_var=generator(lr_var)

                results.append(sr_var.to('cpu'))
                print('Processing Image: %d Part: %d / %d'
                      % (batch + 1, idx + 1, iters), end='\r')
                sys.stdout.flush()

            results = torch.cat(results, dim=0)
            sr_img = tensor_merge(results, hr_size, opt.psize * opt.scala, opt.overlap * opt.scala)


        imname = '%s_X%d.png' % (im_name.split('.')[0], opt.scala)
        img_save_dir = os.path.join(opt.result_dir,'%d/test_SR'%rate)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        # print("*** sr img shape ",sr_img.shape ,sr_var.shape)
        sr_var=sr_img

        if opt.save_results:
            im = to_pil_image(torch.clamp(sr_var[0].cpu() / opt.rgb_range, min=0.0, max=1.0))
            im.save(os.path.join(img_save_dir, im_name))
            # print('[%d/%d] saving to: %s' % (batch + 1, len(test_loader), os.path.join(opt.result_dir, im_name)))


        if opt.has_GT:
            psnr_single = YCbCr_psnr(sr_var.to(device), hr.to(device), scale=opt.scala, peak=opt.rgb_range)
            with torch.no_grad():
                ssim_single = pytorch_ssim.ssim(sr_var.to(device) / opt.rgb_range, hr.to(device) / opt.rgb_range).item()
            # print('%s PSNR: %.4f SSIM: %.4f LPIPS: %.4f' % (im_name, psnr_single, ssim_single, LPIPS_single))
            psnr_sum += psnr_single
            ssim_sum += ssim_single
        sys.stdout.flush()

    if opt.has_GT:
        psnr_sum /= len(test_loader)
        ssim_sum /= len(test_loader)
        psnr_list.append(psnr_sum)
        ssim_list.append(ssim_sum)
        print('-----------\nAve: PSNR: %.4f SSIM: %.4f \n-----------' % (psnr_sum, ssim_sum))
        log_f.write('-----------\nAve: PSNR: %.4f SSIM: %.4f \n-----------' % (psnr_sum, ssim_sum))
        sys.stdout.flush()

#
print("*** PSNR 1: {:.2f}, 2: {:.2f}, 4: {:.2f}, 6: {:.2f}, 8: {:.2f}".format(*psnr_list))
log_f.write("*** PSNR 1: {:.2f}, 2: {:.2f}, 4: {:.2f}, 6: {:.2f}, 8: {:.2f}".format(*psnr_list))
print("*** SSIM 1: {:.3f}, 2: {:.3f}, 4: {:.3f}, 6: {:.3f}, 8: {:.3f}".format(*ssim_list))
log_f.write("*** SSIM 1: {:.3f}, 2: {:.3f}, 4: {:.3f}, 6: {:.3f}, 8: {:.3f}".format(*ssim_list))
#
