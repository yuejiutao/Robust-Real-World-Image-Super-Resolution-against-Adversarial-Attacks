import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

import argparse
import sys
import os
import importlib
import datetime
import tqdm
from tensorboardX import SummaryWriter
sys.path.append('../')
sys.path.append('./modules')

from TorchTools.DataTools.DataSets import SRDataList, SRDataListAC, RealPairDatasetAC, MixRealBicDataset, \
    MixRealBicDatasetAC,RealPairDatasetTest
from TorchTools.Functions.Metrics import psnr
from TorchTools.Functions.Metrics import PSNR, psnr, YCbCr_psnr
import pytorch_ssim
from TorchTools.Functions.functional import tensor_block_cat
from TorchTools.LogTools.logger import Logger
from TorchTools.DataTools.Loaders import to_pil_image, to_tensor
from TorchTools.TorchNet.tools import calculate_parameters, load_weights
from TorchTools.TorchNet.Losses import get_content_loss, TVLoss, VGGFeatureExtractor, contextual_Loss, L1_loss, GW_loss
from TorchTools.DataTools.DataSets import RealPairDataset
from TorchTools.TorchNet.GaussianKernels import random_batch_gaussian_noise_param
import block as B
import warnings
import numpy as np
import cv2
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


warnings.filterwarnings("ignore")
local_test = False

# Load Config File
parser = argparse.ArgumentParser()
parser.add_argument('--runlog',type=str,default='runs/train/tmp')
parser.add_argument('--pretrain',type=str,default='')
parser.add_argument('--exp_name',type=str,default='train/tmp')
parser.add_argument('--model',type=str,default='CDC_MC')
parser.add_argument('--n_HG', type=int, default=6, help='number of feature maps')
parser.add_argument('--config_file', type=str, default='./options/realSR_HGSR_MSHR.py', help='')
parser.add_argument('--train_file', type=str, default='adv_train.py', help='')
parser.add_argument('--gpus', type=int, default=1, help='')
parser.add_argument('--gpu_idx', type=str, default="", help='')
parser.add_argument('--generatorLR', type=float, default=2e-4, help='learning rate for SR generator')
parser.add_argument('--nb_iter' ,type=int ,default=3)
parser.add_argument('--rate',type=float,default=8.0)
parser.add_argument('--test_dataroot', type=str, default='/data1/yuejiutao/Data/RealSR/x4')

args = parser.parse_args()
module_name = os.path.basename(args.config_file).split('.')[0]
config = importlib.import_module('options.' + module_name)
print('Load Config From: %s' % module_name)

# Set Params
opt = config.parse_config(local_test)
use_cuda = opt.use_cuda
opt.config_file = args.config_file
opt.train_file = args.train_file
opt.gpus = args.gpus
device = torch.device('cuda') if use_cuda else torch.device('cpu')
gt_one = torch.ones(opt.batch_size).cuda()
gt_zero = torch.zeros(opt.batch_size).cuda()

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
        generator = model.CDC_MC(in_nc=opt.in_ch, out_nc=opt.out_ch, upscale=opt.scala,
                                 nf=64, res_type=opt.res_type, n_mid=2,
                                 n_HG=opt.n_HG, inter_supervis=opt.inter_supervis,
                                 mscale_inter_super=opt.mscale_inter_super)
    return generator

def get_map(hr):
    hr_Y = (0.257 * hr[:, :1, :, :] + 0.564 * hr[:, 1:2, :, :] + 0.098 * hr[:, 2:, :, :] + 16 / 255.0) * 255.0
    map_corner = hr_Y.new(hr_Y.shape).fill_(0)
    map_edge = hr_Y.new(hr_Y.shape).fill_(0)
    map_flat = hr_Y.new(hr_Y.shape).fill_(0)
    hr_Y_numpy = np.transpose(hr_Y.numpy(), (0, 2, 3, 1))
    for i in range(hr_Y_numpy.shape[0]):
        dst = cv2.cornerHarris(hr_Y_numpy[i, :, :, 0], 3, 3, 0.04)
        thres1 = 0.01 * dst.max()
        thres2 = -0.001 * dst.max()
        map_corner[i, :, :, :] = torch.from_numpy(np.float32(dst > thres1))
        map_edge[i, :, :, :] = torch.from_numpy(np.float32(dst < thres2))
        map_flat[i, :, :, :] = torch.from_numpy(np.float32((dst > thres2) & (dst < thres1)))
    map_corner = map_corner.to(device)
    map_edge = map_edge.to(device)
    map_flat = map_flat.to(device)
    coe_list = []
    coe_list.append(map_flat)
    coe_list.append(map_edge)
    coe_list.append(map_corner)
    return coe_list
#############################################################################################



try:
    opt.need_hr_down
    need_hr_down = opt.need_hr_down
except:
    need_hr_down = False
opt.scale = (opt.scala, opt.scala)

print("*** opt.exp_name ",opt.exp_name)

logger = Logger(opt.exp_name, opt.exp_name, opt)
try:
    epoch_idx = int(opt.pretrain.split('_')[-1].split('.')[0])
except:
    epoch_idx = 0
# epoch_idx = 0

# Prepare Dataset
transform = transforms.Compose([transforms.RandomCrop(opt.size * opt.scala)])
if not opt.bic:
    if opt.mix_bic_real:
        dataset = MixRealBicDatasetAC(opt.dataroot, lr_patch_size=opt.size, scala=opt.scala,
                                      mode='Y' if opt.in_ch == 1 else 'RGB', train=True, real_rate=opt.real_rate)
    else:
        dataset = RealPairDataset(opt.dataroot, lr_patch_size=opt.size, scala=opt.scala,
                                  mode='Y' if opt.in_ch == 1 else 'RGB', train=True,
                                  need_hr_down=need_hr_down, rgb_range=opt.rgb_range,
                                  multiHR=True if opt.model == 'HGSR-MHR' and opt.inter_supervis else False)
    test_dataset = RealPairDataset(opt.test_dataroot, lr_patch_size=opt.size, scala=opt.scala,
                                   mode='Y' if opt.in_ch == 1 else 'RGB', train=False,
                                   need_hr_down=need_hr_down, rgb_range=opt.rgb_range)
else:
    dataset = SRDataList(opt.dataroot, transform=transform, lr_patch_size=opt.size, scala=opt.scala,
                         mode='Y' if opt.in_ch == 1 else 'RGB', train=True)
    test_dataset = SRDataList(opt.test_dataroot, lr_patch_size=opt.size, scala=opt.scala,
                              mode='Y' if opt.in_ch == 1 else 'RGB', train=False)
loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers, drop_last=True)
batches = len(loader)

generator=get_model(opt)
print('%s Created, Parameters: %d' % (generator.__class__.__name__, calculate_parameters(generator)))
# print(generator)
generator = load_weights(generator, opt.pretrain, opt.gpus, init_method='kaiming', scale=0.1, just_weight=False,
                         strict=True)
# generator = load_weights(generator, opt.pretrain, opt.gpus, init_method='kaiming', scale=0.1, strict=False)
generator = generator.to(device)

# Loss Function: L1 + (CX) + (VGG) + (TV)
print('...Initial Loss Criterion...')
content_criterion = get_content_loss(opt.loss, nn_func=False, use_cuda=use_cuda)
test_content_criterion=nn.MSELoss().cuda()

# Init Optim
# optim_ = optim.Adam(generator.parameters(), lr=opt.generatorLR)
optim_ = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=opt.generatorLR)
scheduler = MultiStepLR(optim_, milestones=opt.decay_step, gamma=0.5)
opt.epochs = int(opt.decay_step[-1] // len(loader)) - epoch_idx

for _ in range(epoch_idx *2 * len(loader)):
    scheduler.step()
writer = SummaryWriter(opt.runlog)

print("*** model ",opt.model)
print("*** exp_name ",opt.exp_name)
print("*** opt.rate ",opt.rate)
print("*** opt.epsmax ",opt.clip_eps_max)
print("*** opt.rgb_range ",opt.rgb_range)
print("*** opt.epochs ",opt.epochs)
print("*** opt.test_interval",opt.test_interval)
print("*** learning rate ",scheduler.get_last_lr())


for epoch in range(opt.epochs):

    ''' testing and saving '''
    if epoch % opt.test_interval == 0:

        if epoch % opt.test_interval == 0:
            logger.save('%s_X%d_%d.pth'
                        % (opt.model, opt.scala, epoch + epoch_idx + opt.test_interval), generator, optim_)

        psnr_sum = 0; ssim_sum = 0
        psnr2_sum = 0; ssim2_sum =0
        generator.eval()

        if('EDSR' in opt.model or 'RCAN' in opt.model):
            testrate=8.0; testclip_min=0.0; testclip_max=255.0; testclip_eps_min=-testrate; testclip_eps_max=testrate
        else:
            testrate=8.0/255.0; testclip_min=0.0; testclip_max=1.0; testclip_eps_min=-testrate; testclip_eps_max=testrate

        for batch, data in enumerate(test_loader):
            lr = data['LR']
            hr = data['HR']
            lr_var = lr.to(device)
            hr_var = hr.to(device)
            # print("")\
            test_nb_iter=11
            global_noise_data = torch.zeros_like(lr_var).to(device)
            for nb in range(test_nb_iter):
                noise_batch = Variable(global_noise_data[0:lr_var.size(0)], requires_grad=True).to(device)
                input = lr_var + noise_batch
                input = Variable(input, requires_grad=True)
                loss = 0.

                if opt.model=='CDC_MC':
                    sr_var, _, _, Flag = generator(input)
                    sr_var = sr_var[-1]
                else:
                    sr_var = generator(input)

                loss = test_content_criterion(sr_var, hr_var)
                optim_.zero_grad()

                pert_grad = 0
                loss.backward()
                pert_grad = input.grad.data

                step_size = testrate / (test_nb_iter-1.)
                pert = step_size * torch.sign(pert_grad)

                adv_x = input.detach() + pert
                tmplr = lr_var.detach()
                pertubation = torch.clamp(adv_x - tmplr, testclip_eps_min, testclip_eps_max)
                global_noise_data[0:input.size(0)] = pertubation.data
                global_noise_data.clamp_(testclip_eps_min, testclip_eps_max)

                if ((not opt.no_HR) and (nb == 0 or nb == test_nb_iter-1)):
                    sr_loss = content_criterion(sr_var.detach(), hr_var)
                    psnr_single = YCbCr_psnr(sr_var.detach(), hr_var, scale=opt.scala, peak=opt.rgb_range)

                    if nb == 0:
                        psnr_sum += psnr_single
                    elif nb == test_nb_iter-1:
                        psnr2_sum += psnr_single

                    with torch.no_grad():
                        ssim_single = pytorch_ssim.ssim(sr_var.detach() / opt.rgb_range,
                                                        hr_var / opt.rgb_range).item()
                        if nb == 0:
                            ssim_sum += ssim_single
                        elif nb == test_nb_iter-1:
                            ssim2_sum += ssim_single

        if not opt.no_HR:
            test_info = 'test: psnr: %.4f psnr2: %.4f  srloss: %.4f ' % (psnr_sum / len(test_dataset),psnr2_sum / len(test_dataset), sr_loss.item())
            logger.print_log(test_info, with_time=False)
            writer.add_scalar('psnr', psnr_sum / len(test_dataset), global_step=epoch)
            writer.add_scalar('ssim', ssim_sum / len(test_dataset), global_step=epoch)
            writer.add_scalar('psnr2', psnr2_sum / len(test_dataset), global_step=epoch)
            writer.add_scalar('ssim2', ssim2_sum / len(test_dataset), global_step=epoch)

        else:
            # print("testing...")
            logger.print_log('testing...', with_time=False)



    ''' training '''
    start = datetime.datetime.now()
    generator.train()
    train_loss = 0

    for batch, data in enumerate(loader):
        lr = data['LR']
        hr = data['HR']
        lr_var = lr.to(device)
        hr_var = hr.to(device)

        if(('CDC' in opt.model) or ('HGSR-MHR'in opt.model)):
            coe_list = get_map(hr)

        # global global_noise_data
        global_noise_data = torch.zeros([opt.batch_size, 3, opt.size, opt.size]).to(device)
        for nb in range(opt.nb_iter):
            noise_batch = Variable(global_noise_data[0:lr_var.size(0)], requires_grad=True).to(device)
            input = lr_var + noise_batch
            input = Variable(input,requires_grad=True)
            train_info = '[%03d/%03d][%04d/%04d] '
            train_info_tuple = [epoch + 1, opt.epochs, batch + 1, batches]
            loss=0

            if opt.model=='CDC_MC':
                if(nb==0):
                    sr_var, _, _,Flag = generator(input, train=True, rate=0, gt=gt_zero)
                else:
                    sr_var, _, _,Flag = generator(input, train=True, rate=opt.rate, gt=gt_one)
            else:
                sr_var = generator(input)


            if (('CDC' in opt.model) or ('HGSR-MHR'in opt.model)):
                sr_loss = 0
                for i in range(len(sr_var)):
                    if i != len(sr_var) - 1:
                        coe = coe_list[i]
                        single_srloss = opt.inte_loss_weight[i] * content_criterion(coe * sr_var[i], coe * hr_var)
                    else:
                        single_srloss = opt.inte_loss_weight[i] * GW_loss(sr_var[i], hr_var)
                    sr_loss += single_srloss
                    train_info += ' H%d: %.4f'
                    train_info_tuple.append(i)
                    train_info_tuple.append(single_srloss.item())
            else:
                sr_loss = content_criterion(sr_var, hr_var)


            loss = sr_loss * opt.sr_lambda
            train_loss += loss
            if(nb==0 or nb==opt.nb_iter-1):
                train_info += ' tot: %.4f'
                train_info_tuple.append(loss.item())

            optim_.zero_grad()
            pert_grad=0
            loss.backward()
            pert_grad = input.grad.data

            if nb==0 or nb==opt.nb_iter-1:
                optim_.step()
                scheduler.step()

            step_size = opt.rate / (opt.nb_iter-1)
            pert = step_size * torch.sign(pert_grad)

            adv_x = input.detach() + pert
            tmplr = lr_var.detach()
            pertubation = torch.clamp(adv_x - tmplr, opt.clip_eps_min, opt.clip_eps_max)
            global_noise_data[0:input.size(0)] = pertubation.data
            global_noise_data.clamp_(opt.clip_eps_min, opt.clip_eps_max)

            if(nb==0 or nb==opt.nb_iter-1):
                logger.print_log(train_info % tuple(train_info_tuple), with_time=False)

            iter_n = batch + epoch * len(loader)
    
    if epoch % opt.test_interval==0:
        writer.add_scalar('train_loss', train_loss.item() / (len(loader)* opt.nb_iter), global_step=epoch)
    end = datetime.datetime.now()
    running_lr = scheduler.get_last_lr()
    logger.print_log(' epoch: [%d/%d] elapse: %s lr: %.6f'
                     % (epoch + 1, opt.epochs, str(end - start)[:-4], running_lr[0]))

