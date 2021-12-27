import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from . import block as B
import block as B
import pdb

import sys
sys.path.append('../')
sys.path.append('../../../')

from TorchTools.Functions.visual_fea import *
import kornia
import torch_dct as mydct


class ResidualBlock(nn.Module):
    """
    Residual BLock For SR without Norm Layer
    conv 1*1
    conv 3*3
    conv 1*1
    """

    def __init__(self, ch_in=64, ch_out=128, in_place=True):
        super(ResidualBlock, self).__init__()

        conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        conv2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        conv3 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        relu1 = nn.LeakyReLU(0.2, inplace=in_place)
        relu2 = nn.LeakyReLU(0.2, inplace=in_place)

        self.res = nn.Sequential(conv1, relu1, conv2, relu2, conv3)
        if ch_in != ch_out:
            self.identity = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        else:
            def identity(tensor):
                return tensor

            self.identity = identity

    def forward(self, x):
        res = self.res(x)
        x = self.identity(x)
        return torch.add(x, res)


class ResidualInceptionBlock(nn.Module):
    """
    Residual Inception BLock For SR without Norm Layer
    conv 1*1  conv 1*1  conv 1*1
              conv 3*3  conv 3*3
                        conv 3*3
              concat
              conv 1*1
    """

    def __init__(self, ch_in=64, ch_out=128, in_place=True):
        super(ResidualInceptionBlock, self).__init__()

        conv1_1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        relu1_1 = nn.LeakyReLU(0.2, inplace=in_place)

        conv2_1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        conv2_2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        relu2_2 = nn.LeakyReLU(0.2, inplace=in_place)

        conv3_1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0, bias=False)
        conv3_2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        relu3_2 = nn.LeakyReLU(0.2, inplace=in_place)
        conv3_3 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        relu3_3 = nn.LeakyReLU(0.2, inplace=in_place)

        self.res1 = nn.Sequential(conv1_1, relu1_1)
        self.res2 = nn.Sequential(conv2_1, conv2_2, relu2_2)
        self.res3 = nn.Sequential(conv3_1, conv3_2, relu3_2, conv3_3, relu3_3)
        self.filter_concat = nn.Conv2d(in_channels=3 * ch_out, out_channels=ch_out, kernel_size=1, stride=1, padding=0,
                                       bias=False)
        if ch_in != ch_out:
            self.identity = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        else:
            def identity(tensor):
                return tensor

            self.identity = identity

    def forward(self, x):
        res1 = self.res1(x)
        res2 = self.res2(x)
        res3 = self.res3(x)
        res = torch.cat((res1, res2, res3), dim=1)
        res = self.filter_concat(res)
        x = self.identity(x)
        return torch.add(x, res)


class TopDownBlock(nn.Module):
    """
    Top to Down Block for HourGlass Block
    Consist of ConvNet Block and Pooling
    """

    def __init__(self, ch_in=64, ch_out=64, res_type='res'):
        super(TopDownBlock, self).__init__()
        if res_type == 'rrdb':
            self.res_block = B.RRDB(ch_in, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                    norm_type=None, act_type='leakyrelu', mode='CNA')
        else:
            self.res_block = ResidualBlock(ch_in=ch_in, ch_out=ch_out)
            # self.res_block = ResidualInceptionBlock(ch_in=ch_in, ch_out=ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.res_block(x)
        return self.pool(x), x


class BottomUpBlock(nn.Module):
    """
    Bottom Up Block for HourGlass Block
    Consist of ConvNet Block and Upsampling Block
    """

    def __init__(self, ch_in=64, ch_out=64, res_type='res'):
        super(BottomUpBlock, self).__init__()
        if res_type == 'rrdb':
            self.res_block = B.RRDB(ch_in, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                    norm_type=None, act_type='leakyrelu', mode='CNA')
        else:
            self.res_block = ResidualBlock(ch_in=ch_in, ch_out=ch_out)
            # self.res_block = ResidualInceptionBlock(ch_in=ch_in, ch_out=ch_out)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, res):
        x = self.upsample(x)
        return self.res_block(x + res)


class HourGlassBlock(nn.Module):
    """
    Hour Glass Block for SR Model
    """

    def __init__(self, res_type='res', n_mid=2, n_tail=2):
        super(HourGlassBlock, self).__init__()
        self.n_tail = n_tail

        self.down1 = TopDownBlock(64, 128, res_type=res_type)
        self.down2 = TopDownBlock(128, 128, res_type=res_type)
        self.down3 = TopDownBlock(128, 256, res_type=res_type)
        self.down4 = TopDownBlock(256, 256, res_type=res_type)

        res_block = []
        for i in range(n_mid):
            res_block.append(ResidualBlock(256, 256))
        self.mid_res = nn.Sequential(*res_block)

        self.skip_conv0 = ResidualBlock(ch_in=64, ch_out=64)
        self.skip_conv1 = ResidualBlock(ch_in=128, ch_out=128)
        self.skip_conv2 = ResidualBlock(ch_in=128, ch_out=128)
        self.skip_conv3 = ResidualBlock(ch_in=256, ch_out=256)
        self.skip_conv4 = ResidualBlock(ch_in=256, ch_out=256)

        self.up1 = BottomUpBlock(256, 256, res_type=res_type)
        self.up2 = BottomUpBlock(256, 128, res_type=res_type)
        self.up3 = BottomUpBlock(128, 128, res_type=res_type)
        self.up4 = BottomUpBlock(128, 64, res_type=res_type)

        if n_tail != 0:
            tail_block = []
            for i in range(n_tail):
                tail_block.append(ResidualInceptionBlock(64, 64))
            self.tail = nn.Sequential(*tail_block)

    def forward(self, x):
        out, res1 = self.down1(x)

        out, res2 = self.down2(out)
        out, res3 = self.down3(out)
        out, res4 = self.down4(out)

        out = self.mid_res(out)

        out = self.up1(out, self.skip_conv4(res4))
        out = self.up2(out, self.skip_conv3(res3))
        out = self.up3(out, self.skip_conv2(res2))
        out = self.up4(out, self.skip_conv1(res1))
        out_inter = self.skip_conv0(x) + out

        if self.n_tail != 0:
            out = self.tail(out_inter)
        else:
            out = out_inter
        # Change to Residul Structure
        return out, out_inter



class MaskBlock(nn.Module):
    """
    Bottom Up Block for HourGlass Block
    Consist of ConvNet Block and Upsampling Block
    """
    def __init__(self,down=0.43,up=0.5,SFM_center_radius_perc=0.70, SFM_center_sigma_perc=0.15):
        super(MaskBlock, self).__init__()
        self.SFM_center_radius_perc=SFM_center_radius_perc
        self.SFM_center_sigma_perc=SFM_center_sigma_perc
        self.down=down
        self.up=up

    def circular_random_drop_mask(self,w=256, h=256, SFM_center_radius_perc=0.70, SFM_center_sigma_perc=0.15):
        radius = np.sqrt(w * w + h * h)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        Y, X = torch.meshgrid(torch.linspace(0, w - 1, w).cuda(), torch.linspace(0, h - 1, h).cuda())
        D = torch.sqrt(X * X + Y * Y)

        lrdown = random.uniform(self.down, self.up)
        lrdown = lrdown * radius

        mask0 = torch.ones((w, h), dtype=torch.int, device='cuda')
        mask0[D > lrdown] = 0
        # print(type(mask0))
        D = D / torch.max(D)
        mask = torch.ones((w, h), dtype=torch.int, device='cuda') - torch.bernoulli(D).int()
        mask = mask | mask0

        return mask

    def forward(self, x):
        shape=x.shape
        mask=self.circular_random_drop_mask(shape[-2],shape[-1])

        dctx=mydct.dct_2d(x,norm='ortho')
        dctx=dctx*mask
        output=mydct.idct_2d(dctx,norm='ortho')

        return output



def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class BasicBlockNL(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlockNL, self).__init__(*m)

class NonLocalDenoise(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=default_conv):
        super(NonLocalDenoise, self).__init__()
        self.conv_match1 = BasicBlockNL(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = BasicBlockNL(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = input

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).reshape((N, H * W, C))
        x_embed_2 = x_embed_2.reshape(N, C, H * W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)

        x_assembly = x_assembly.reshape(N, -1, H * W).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly)
        x_final=x_final.permute(0, 2, 1).view(N, -1, H, W)

        return x_final




class MaskHgBlock(nn.Module):
    """
    Hour Glass Block for SR Model
    """

    def __init__(self, res_type='res', n_mid=2, n_tail=2):
        super(MaskHgBlock, self).__init__()
        self.n_tail = n_tail
        self.pre_mask = MaskBlock()

        self.down1 = TopDownBlock(64, 128, res_type=res_type)
        self.down2 = TopDownBlock(128, 128, res_type=res_type)
        self.down3 = TopDownBlock(128, 256, res_type=res_type)
        self.down4 = TopDownBlock(256, 256, res_type=res_type)

        res_block = []
        for i in range(n_mid):
            res_block.append(ResidualBlock(256, 256))
        self.mid_res = nn.Sequential(*res_block)

        self.skip_conv0 = ResidualBlock(ch_in=64, ch_out=64)
        self.skip_conv1 = ResidualBlock(ch_in=128, ch_out=128)
        self.skip_conv2 = ResidualBlock(ch_in=128, ch_out=128)
        self.skip_conv3 = ResidualBlock(ch_in=256, ch_out=256)
        self.skip_conv4 = ResidualBlock(ch_in=256, ch_out=256)

        self.up1 = BottomUpBlock(256, 256, res_type=res_type)
        self.up2 = BottomUpBlock(256, 128, res_type=res_type)
        self.up3 = BottomUpBlock(128, 128, res_type=res_type)
        self.up4 = BottomUpBlock(128, 64, res_type=res_type)

        self.post_mask = MaskBlock()

        if n_tail != 0:
            tail_block = []
            for i in range(n_tail):
                tail_block.append(ResidualInceptionBlock(64, 64))
            self.tail = nn.Sequential(*tail_block)

    def forward(self, x, Flag=False):

        if(Flag):
            x=self.pre_mask(x)

        out, res1 = self.down1(x)
        out, res2 = self.down2(out)
        out, res3 = self.down3(out)
        out, res4 = self.down4(out)

        out = self.mid_res(out)

        out = self.up1(out, self.skip_conv4(res4))
        out = self.up2(out, self.skip_conv3(res3))
        out = self.up3(out, self.skip_conv2(res2))
        out = self.up4(out, self.skip_conv1(res1))
        out_inter = self.skip_conv0(x) + out

        # Fea_vis_dct.vis_fea_dct(out_inter,'hg{}_xout.png'.format(hgid))
        # out_inter=self.post_mask(out_inter)
        if(Flag):
            out_inter=self.post_mask(out_inter)

        # Fea_vis_dct.vis_fea_dct(out_inter,'hg{}_xout_mask.png'.format(hgid))

        if self.n_tail != 0:
            out = self.tail(out_inter)
        else:
            out = out_inter
        # Change to Residul Structure
        return out, out_inter


class CDC(nn.Module):
    """
    Hour Glass SR Model, Use Mutil-Scale Label(HR_down_Xn) Supervision.
    """

    def __init__(self, in_nc=3, out_nc=3, upscale=4, nf=64, res_type='res', n_mid=2, n_tail=2, n_HG=6,
                 act_type='leakyrelu', inter_supervis=True, mscale_inter_super=False, share_upsample=False):
        super(CDC, self).__init__()

        self.n_HG = n_HG
        self.inter_supervis = inter_supervis
        if upscale == 3:
            ksize = 3
        else:
            ksize = 1

        self.conv_in = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        def make_upsample_block(upscale=4, in_ch=64, out_nc=3, kernel_size=3):
            n_upscale = 1 if upscale == 3 else int(math.log(upscale, 2))
            LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
            HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='leakyrelu')
            HR_conv1 = B.conv_block(nf, out_nc, kernel_size=kernel_size, norm_type=None, act_type=None)
            if upscale == 1:
                return nn.Sequential(LR_conv, HR_conv0, HR_conv1)
            elif upscale == 3:
                upsampler = B.upconv_blcok(nf, nf, 3, act_type=act_type)
            else:
                upsampler = [B.upconv_blcok(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            return nn.Sequential(LR_conv, *upsampler, HR_conv0, HR_conv1)

        # Actually, the ksize can be any size for all scales
        self.flat_map = make_upsample_block(upscale=upscale, kernel_size=ksize)
        self.edge_map = make_upsample_block(upscale=upscale, kernel_size=ksize)
        self.corner_map = make_upsample_block(upscale=upscale, kernel_size=ksize)

        self.upsample_flat = make_upsample_block(upscale=upscale)
        self.upsample_edge = make_upsample_block(upscale=upscale)
        self.upsample_corner = make_upsample_block(upscale=upscale)

        for i in range(n_HG):
            if i != n_HG - 1:
                HG_block = HourGlassBlock(res_type=res_type, n_mid=n_mid, n_tail=n_tail)
            else:
                HG_block = HourGlassBlock(res_type=res_type, n_mid=n_mid, n_tail=0)
            setattr(self, 'HG_%d' % i, HG_block)

    def forward(self, x):
        x = self.conv_in(x)

        SR_map = []
        result = []
        out = x

        # Multi-Scale supervise, [2, 2, 2] for 6, [2, 3, 3] for 8
        super_block_idx = [1, self.n_HG // 2, self.n_HG - 1]

        for i in range(self.n_HG):
            out, out_inter = getattr(self, 'HG_%d' % i)(out)
            if i in super_block_idx:
                if i == self.n_HG - 1:
                    sr_feature = out.mul(0.2) + x
                else:
                    sr_feature = out_inter

                if super_block_idx.index(i) == 0:
                    srout_flat = self.upsample_flat(sr_feature)
                    flat_map = self.flat_map(sr_feature)
                    result.append(srout_flat)
                elif super_block_idx.index(i) == 1:
                    srout_edge = self.upsample_edge(sr_feature)
                    edge_map = self.edge_map(sr_feature)
                    result.append(srout_edge)
                elif super_block_idx.index(i) == 2:
                    srout_corner = self.upsample_corner(sr_feature)
                    corner_map = self.corner_map(sr_feature)
                    result.append(srout_corner)

                    flat_r, flat_g, flat_b = flat_map.split(split_size=1, dim=1)
                    edge_r, edge_g, edge_b = edge_map.split(split_size=1, dim=1)
                    corner_r, corner_g, corner_b = corner_map.split(split_size=1, dim=1)
                    r_map = torch.cat((flat_r, edge_r, corner_r), dim=1)
                    g_map = torch.cat((flat_g, edge_g, corner_g), dim=1)
                    b_map = torch.cat((flat_b, edge_b, corner_b), dim=1)
                    r_map = F.softmax(r_map, dim=1)
                    g_map = F.softmax(g_map, dim=1)
                    b_map = F.softmax(b_map, dim=1)
                    flat_r, edge_r, corner_r = r_map.split(split_size=1, dim=1)
                    flat_g, edge_g, corner_g = g_map.split(split_size=1, dim=1)
                    flat_b, edge_b, corner_b = b_map.split(split_size=1, dim=1)
                    flat_map = torch.cat((flat_r, flat_g, flat_b), dim=1)
                    edge_map = torch.cat((edge_r, edge_g, edge_b), dim=1)
                    corner_map = torch.cat((corner_r, corner_g, corner_b), dim=1)
                    srout = flat_map * srout_flat + edge_map * srout_edge + corner_map * srout_corner
                    result.append(srout)
                    SR_map.append(torch.mean(flat_map, dim=1, keepdim=True))
                    SR_map.append(torch.mean(edge_map, dim=1, keepdim=True))
                    SR_map.append(torch.mean(corner_map, dim=1, keepdim=True))
                # result.append(sr_feature[:,0:1,:,:])

        return result, SR_map



class MaskClassifier(nn.Module):
    def __init__(self,inc=64,outc=64):
        super(MaskClassifier, self).__init__()
        self.gt=0

        self.unfold=torch.nn.Unfold(kernel_size=3, dilation=1, padding=0, stride=3)
        self.avgpool=nn.AvgPool1d(kernel_size=9)

        self.fcall = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256,2)
        )

        self.criterion =nn.CrossEntropyLoss().cuda()
        # self.Loss=nn.MSELoss()

    def forward(self, x ,gt):
        self.gt=gt

        dctx=mydct.dct_2d(x,norm='ortho')
        dctx= torch.mean(dctx,dim=1,keepdim=True)

        eps=torch.ones_like(dctx)
        eps[:,:,:,:]=1e-6

        dctx = torch.where(dctx == 0.0, eps, dctx)
        dctx = torch.log(torch.abs(dctx))
        if(dctx.shape[2]!=48):
            dctx=F.adaptive_avg_pool2d(dctx,output_size=48)
        

        mid = self.unfold(dctx)
        mid = mid.permute(0,2,1)
        mid = self.avgpool(mid)
        mid = mid.squeeze(dim=2)
        mid = self.fcall(mid)
        loss = self.criterion(mid,self.gt.long())
        mid_s = F.softmax(mid,dim=1)
        ans = mid_s[:,0]
        jud = torch.sum(ans)
        flag = jud<(mid.shape[0]*0.9)

        return flag , loss



class CDC_MC(nn.Module):
    """
    Hour Glass SR Model, Use Mutil-Scale Label(HR_down_Xn) Supervision.
    """

    def __init__(self, in_nc=3, out_nc=3, upscale=4, nf=64, res_type='res', n_mid=2, n_tail=2, n_HG=6,
                 act_type='leakyrelu', inter_supervis=True, mscale_inter_super=False, share_upsample=False):
        super(CDC_MC, self).__init__()

        self.n_HG = n_HG
        self.inter_supervis = inter_supervis
        if upscale == 3:
            ksize = 3
        else:
            ksize = 1

        self.preclass = MaskClassifier()
        self.premask = MaskBlock()
        self.conv_in = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        def make_upsample_block(upscale=4, in_ch=64, out_nc=3, kernel_size=3):
            n_upscale = 1 if upscale == 3 else int(math.log(upscale, 2))
            LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
            HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='leakyrelu')
            HR_conv1 = B.conv_block(nf, out_nc, kernel_size=kernel_size, norm_type=None, act_type=None)
            if upscale == 1:
                return nn.Sequential(LR_conv, HR_conv0, HR_conv1)
            elif upscale == 3:
                upsampler = B.upconv_blcok(nf, nf, 3, act_type=act_type)
            else:
                upsampler = [B.upconv_blcok(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            return nn.Sequential(LR_conv, *upsampler, HR_conv0, HR_conv1)

        # Actually, the ksize can be any size for all scales
        self.flat_map = make_upsample_block(upscale=upscale, kernel_size=ksize)
        self.edge_map = make_upsample_block(upscale=upscale, kernel_size=ksize)
        self.corner_map = make_upsample_block(upscale=upscale, kernel_size=ksize)

        self.upsample_flat = make_upsample_block(upscale=upscale)
        self.upsample_edge = make_upsample_block(upscale=upscale)
        self.upsample_corner = make_upsample_block(upscale=upscale)

        for i in range(n_HG):
            if i != n_HG - 1:
                HG_block = MaskHgBlock(res_type=res_type, n_mid=n_mid, n_tail=n_tail)
            else:
                HG_block = MaskHgBlock(res_type=res_type, n_mid=n_mid, n_tail=0)
            setattr(self, 'HG_%d' % i, HG_block)

    def forward(self, x, train=False, rate=0, gt=0):
        if (train == False):
            gt = torch.zeros(x.shape[0]).cuda()

        if (train):
            Flag, judloss = self.preclass(x, gt)
            if (rate > 0):
                Flag = True
            else:
                Flag = False
        else:
            Flag, judloss = self.preclass(x, gt)

        if (Flag):
            x = self.premask(x)

        x = self.conv_in(x)
        SR_map = []
        result = []
        out = x

        # Multi-Scale supervise, [2, 2, 2] for 6, [2, 3, 3] for 8
        super_block_idx = [1, self.n_HG // 2, self.n_HG - 1]

        for i in range(self.n_HG):
            out, out_inter = getattr(self, 'HG_%d' % i)(out, Flag)
            if i in super_block_idx:
                if i == self.n_HG - 1:
                    sr_feature = out.mul(0.2) + x
                else:
                    sr_feature = out_inter

                if super_block_idx.index(i) == 0:
                    srout_flat = self.upsample_flat(sr_feature)
                    flat_map = self.flat_map(sr_feature)
                    result.append(srout_flat)
                elif super_block_idx.index(i) == 1:
                    srout_edge = self.upsample_edge(sr_feature)
                    edge_map = self.edge_map(sr_feature)
                    result.append(srout_edge)
                elif super_block_idx.index(i) == 2:
                    srout_corner = self.upsample_corner(sr_feature)
                    corner_map = self.corner_map(sr_feature)
                    result.append(srout_corner)

                    flat_r, flat_g, flat_b = flat_map.split(split_size=1, dim=1)
                    edge_r, edge_g, edge_b = edge_map.split(split_size=1, dim=1)
                    corner_r, corner_g, corner_b = corner_map.split(split_size=1, dim=1)
                    r_map = torch.cat((flat_r, edge_r, corner_r), dim=1)
                    g_map = torch.cat((flat_g, edge_g, corner_g), dim=1)
                    b_map = torch.cat((flat_b, edge_b, corner_b), dim=1)
                    r_map = F.softmax(r_map, dim=1)
                    g_map = F.softmax(g_map, dim=1)
                    b_map = F.softmax(b_map, dim=1)
                    flat_r, edge_r, corner_r = r_map.split(split_size=1, dim=1)
                    flat_g, edge_g, corner_g = g_map.split(split_size=1, dim=1)
                    flat_b, edge_b, corner_b = b_map.split(split_size=1, dim=1)
                    flat_map = torch.cat((flat_r, flat_g, flat_b), dim=1)
                    edge_map = torch.cat((edge_r, edge_g, edge_b), dim=1)
                    corner_map = torch.cat((corner_r, corner_g, corner_b), dim=1)
                    srout = flat_map * srout_flat + edge_map * srout_edge + corner_map * srout_corner
                    result.append(srout)
                    SR_map.append(torch.mean(flat_map, dim=1, keepdim=True))
                    SR_map.append(torch.mean(edge_map, dim=1, keepdim=True))
                    SR_map.append(torch.mean(corner_map, dim=1, keepdim=True))
                # result.append(sr_feature[:,0:1,:,:])

        return result, SR_map, judloss, Flag




class HourGlassNetMultiScaleInt(nn.Module):
    """
    Hour Glass SR Model, Use Mutil-Scale Label(HR_down_Xn) Supervision.
    """

    def __init__(self, in_nc=3, out_nc=3, upscale=4, nf=64, res_type='res', n_mid=2, n_tail=2, n_HG=6,
                 act_type='leakyrelu', inter_supervis=True, mscale_inter_super=False, share_upsample=False):
        super(HourGlassNetMultiScaleInt, self).__init__()

        self.n_HG = n_HG
        self.inter_supervis = inter_supervis
        if upscale == 3:
            ksize = 3
        else:
            ksize = 1

        self.conv_in = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        def make_upsample_block(upscale=4, in_ch=64, out_nc=3, kernel_size=3):
            n_upscale = 1 if upscale == 3 else int(math.log(upscale, 2))
            LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
            HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='leakyrelu')
            HR_conv1 = B.conv_block(nf, out_nc, kernel_size=kernel_size, norm_type=None, act_type=None)
            if upscale == 1:
                return nn.Sequential(LR_conv, HR_conv0, HR_conv1)
            elif upscale == 3:
                upsampler = B.upconv_blcok(nf, nf, 3, act_type=act_type)
            else:
                upsampler = [B.upconv_blcok(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            return nn.Sequential(LR_conv, *upsampler, HR_conv0, HR_conv1)

        # Actually, the ksize can be any size for all scales
        self.flat_map = make_upsample_block(upscale=upscale, kernel_size=ksize)
        self.edge_map = make_upsample_block(upscale=upscale, kernel_size=ksize)
        self.corner_map = make_upsample_block(upscale=upscale, kernel_size=ksize)

        self.upsample_flat = make_upsample_block(upscale=upscale)
        self.upsample_edge = make_upsample_block(upscale=upscale)
        self.upsample_corner = make_upsample_block(upscale=upscale)

        for i in range(n_HG):
            if i != n_HG - 1:
                HG_block = HourGlassBlock(res_type=res_type, n_mid=n_mid, n_tail=n_tail)
            else:
                HG_block = HourGlassBlock(res_type=res_type, n_mid=n_mid, n_tail=0)
            setattr(self, 'HG_%d' % i, HG_block)

    def forward(self, x):
        x = self.conv_in(x)
        SR_map = []
        result = []
        out = x

        # Multi-Scale supervise, [2, 2, 2] for 6, [2, 3, 3] for 8
        super_block_idx = [1, self.n_HG // 2, self.n_HG - 1]

        for i in range(self.n_HG):
            out, out_inter = getattr(self, 'HG_%d' % i)(out,i+1)
            if i in super_block_idx:
                if i == self.n_HG - 1:
                    sr_feature = out.mul(0.2) + x
                else:
                    sr_feature = out_inter

                if super_block_idx.index(i) == 0:
                    srout_flat = self.upsample_flat(sr_feature)
                    flat_map = self.flat_map(sr_feature)
                    result.append(srout_flat)
                elif super_block_idx.index(i) == 1:
                    srout_edge = self.upsample_edge(sr_feature)
                    edge_map = self.edge_map(sr_feature)
                    result.append(srout_edge)
                elif super_block_idx.index(i) == 2:
                    srout_corner = self.upsample_corner(sr_feature)
                    corner_map = self.corner_map(sr_feature)
                    result.append(srout_corner)
                    flat_r, flat_g, flat_b = flat_map.split(split_size=1, dim=1)
                    edge_r, edge_g, edge_b = edge_map.split(split_size=1, dim=1)
                    corner_r, corner_g, corner_b = corner_map.split(split_size=1, dim=1)
                    r_map = torch.cat((flat_r, edge_r, corner_r), dim=1)
                    g_map = torch.cat((flat_g, edge_g, corner_g), dim=1)
                    b_map = torch.cat((flat_b, edge_b, corner_b), dim=1)
                    r_map = F.softmax(r_map, dim=1)
                    g_map = F.softmax(g_map, dim=1)
                    b_map = F.softmax(b_map, dim=1)
                    flat_r, edge_r, corner_r = r_map.split(split_size=1, dim=1)
                    flat_g, edge_g, corner_g = g_map.split(split_size=1, dim=1)
                    flat_b, edge_b, corner_b = b_map.split(split_size=1, dim=1)
                    flat_map = torch.cat((flat_r, flat_g, flat_b), dim=1)
                    edge_map = torch.cat((edge_r, edge_g, edge_b), dim=1)
                    corner_map = torch.cat((corner_r, corner_g, corner_b), dim=1)
                    srout = flat_map * srout_flat + edge_map * srout_edge + corner_map * srout_corner
                    result.append(srout)
                    SR_map.append(torch.mean(flat_map, dim=1, keepdim=True))
                    SR_map.append(torch.mean(edge_map, dim=1, keepdim=True))
                    SR_map.append(torch.mean(corner_map, dim=1, keepdim=True))
                # result.append(sr_feature[:,0:1,:,:])

        return result, SR_map