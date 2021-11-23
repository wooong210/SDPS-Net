import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from . import newconv

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = newconv.AugmentedConv(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1, dk=40, dv=4, Nh=4, relative=True, shape=16).cuda()
        self.conv2 = newconv.AugmentedConv(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dk=40, dv=4, Nh=4, relative=True, shape=16).cuda()
        self.conv3 = newconv.AugmentedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dk=40, dv=4, Nh=4, relative=True, shape=16).cuda()
        self.conv4 = newconv.AugmentedConv(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, dk=40, dv=4, Nh=4, relative=True, shape=16).cuda()
        self.conv5 = newconv.AugmentedConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dk=40, dv=4, Nh=4, relative=True, shape=16).cuda()
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = newconv.AugmentedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dk=40, dv=4, Nh=4, relative=True, shape=16).cuda()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat   = out_feat.view(-1)
        return out_feat, [n, c, h, w]

class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}): 
        super(Regressor, self).__init__()
        self.other   = other
        self.deconv1 = newconv.AugmentedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dk=40, dv=4, Nh=4, relative=True, shape=16).cuda()
        self.deconv2 = newconv.AugmentedConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dk=40, dv=4, Nh=4, relative=True, shape=16).cuda()
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class NENet(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(NENet, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def prepareInputs(self, x):
        imgs = torch.split(x[0], 3, 1)
        idx = 1
        if self.other['in_light']: idx += 1
        if self.other['in_mask']:  idx += 1
        dirs = torch.split(x[idx]['dirs'], x[0].shape[0], 0)
        ints = torch.split(x[idx]['intens'], 3, 1)
        
        s2_inputs = []
        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape
            l_dir = dirs[i] if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1)
            l_int = torch.diag(1.0 / (ints[i].contiguous().view(-1)+1e-8))
            img   = imgs[i].contiguous().view(n * c, h * w)
            img   = torch.mm(l_int, img).view(n, c, h, w)
            img_light = torch.cat([img, l_dir.expand_as(img)], 1)
            s2_inputs.append(img_light)
        return s2_inputs

    def forward(self, x):
        inputs = self.prepareInputs(x)
        feats = torch.Tensor()
        for i in range(len(inputs)):
            feat, shape = self.extractor(inputs[i])
            if i == 0:
                feats = feat
            else:
                if self.fuse_type == 'mean':
                    feats = torch.stack([feats, feat], 1).sum(1)
                elif self.fuse_type == 'max':
                    feats, _ = torch.stack([feats, feat], 1).max(1)
        if self.fuse_type == 'mean':
            feats = feats / len(img_split)
        feat_fused = feats
        normal = self.regressor(feat_fused, shape)
        pred = {}
        pred['n'] = normal
        return pred


# CUDA_VISIBLE_DEVICES=0 python main_stage2.py --in_img_num 16 --retrain data/logdir/UPS_Synth_Dataset/CVPR2019/4-17,LCNet,max,adam,cos,ba_h-32,sc_h-128,cr_h-128,in_r-0.0005,no_w-1,di_w-1,in_w-1,in_m-8,di_s-36,in_s-20,in_mask,s1_est_d,s1_est_i,color_aug,int_aug,concat_data/checkpointdir/checkp_20.pth.tar
# CUDA_VISIBLE_DEVICES=0 python main_stage2.py --in_img_num 4 --retrain data/logdir/UPS_Synth_Dataset/CVPR2019/4-19,LCNet,max,adam,cos,ba_h-32,sc_h-128,cr_h-128,in_r-0.0005,no_w-1,di_w-1,in_w-1,in_m-4,di_s-36,in_s-20,in_mask,s1_est_d,s1_est_i,color_aug,int_aug,concat_data/checkpointdir/checkp_20.pth.tar
