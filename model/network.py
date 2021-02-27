import torchvision
import torch.nn as nn
import torch.nn.functional as F
import StackedHourGlass as M
import torch
import torch.utils.data as Data
import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import torchvision.models as models

import sys
import cv2
import dlib
import torchvision
import StackedHourGlass as M
#from network import *
from FAN import *
from torchvision import transforms
from math import floor
from layers import *
from utils import resize,elementwise_mult_cast_int
#from spectral import SpectralNorm

from torch.nn.init import xavier_normal , kaiming_normal
import torchvision.utils as vutils
import time
import warnings
import copy


emci = elementwise_mult_cast_int




### define Generator and Discriminator

## define attention model

class Self_Attn(nn.Module):
    #""" Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        ##"""
        #    inputs :
        #        x : input feature maps( B X C X W X H)
        #    returns :
        #        out : self attention value + input feature
        #        attention: B X N X N (N is Width*Height)
        #"""
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention



class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class ResidualBlockme(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlockme, self).__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, 1, 1),
                        SwitchNorm2d(in_features),
                        nn.ReLU(),
                        nn.Conv2d(in_features, in_features, 3, 1, 1),
                        SwitchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)
        #self.attn = Self_Attn(in_features, 'relu')

    def forward(self, x):
        res= x + self.conv_block(x)
        #resout,p = self.attn(res)
        return res



def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,15,96,96)
    :return:numpy array (N,15,2)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points



five_pts_idx = [ [36,41] , [42,47] , [27,35] , [48,48] , [54,54] ]
def landmarks_68_to_5(x):
    y = []
    for j in range(5):
        y.append( np.mean( x[five_pts_idx[j][0]:five_pts_idx[j][1] + 1] , axis = 0  ) )
    return np.array( y , np.float32)



def process(img, landmarks_5pts):
    batch = {}
    name = ['left_eye','right_eye','nose','mouth']
    patch_size = {
            'left_eye':(40,40),
            'right_eye':(40,40),
            'nose':(40,32),
            'mouth':(48,32),
    }
    landmarks_5pts[3,0] =  (landmarks_5pts[3,0] + landmarks_5pts[4,0]) / 2.0
    landmarks_5pts[3,1] = (landmarks_5pts[3,1] + landmarks_5pts[4,1]) / 2.0

    # crops
    for i in range(4):
        x = floor(landmarks_5pts[i,0])
        y = floor(landmarks_5pts[i,1])
        batch[ name[i] ] = img.crop( (x - patch_size[ name[i] ][0]//2 + 1 , y - patch_size[ name[i] ][1]//2 + 1 , x + patch_size[ name[i] ][0]//2 + 1 , y + patch_size[ name[i] ][1]//2 + 1 ) )

    return batch




class FeaturePredict(nn.Module):
    def __init__(self ,  num_classes , global_feature_layer_dim = 256 , dropout = 0.3):
        super(FeaturePredict,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(global_feature_layer_dim , num_classes )
    def forward(self ,x ,use_dropout):
        if use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class GeneratorResNet(nn.Module):
    def __init__(self):
        super(GeneratorResNet, self).__init__()
       

        self.convvv1 = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1),
                      SwitchNorm2d(128),
                      nn.ReLU()
                     )


        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.convvv2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                       SwitchNorm2d(64),
                       nn.ReLU()
                       )

        self.conv32 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                        SwitchNorm2d(32),
                        nn.ReLU()
                       )

                                     
        self.conv323 = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1),
                                     nn.Tanh())  
        
        # Upsampling layers
        upsampling = []
        for out_features in range(1):
           upsampling += [ nn.Conv2d(64, 256, 3, 1, 1),
                           SwitchNorm2d(256),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.up2 = nn.Sequential(*upsampling)
        
        self.conv64 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                        SwitchNorm2d(32),
                        nn.ReLU()
                       )

                                     
        self.conv643 = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1),
                                     nn.Tanh())  
        
        # Residual blocks
        res_blocks = []
        for _ in range(2):
            res_blocks.append(ResidualBlockme(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.att64 = Self_Attn(64, 'relu')
        #~ self.convvv3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                        #~ SwitchNorm2d(64),
                        #~ nn.ReLU()
                       #~ )


        # Upsampling layers
        upsampling = []
        for out_features in range(1):
           upsampling += [ nn.Conv2d(64, 256, 3, 1, 1),
                           SwitchNorm2d(256),
                           nn.PixelShuffle(upscale_factor=2),
                           nn.ReLU(inplace=True)]
        self.up3 = nn.Sequential(*upsampling)

        self.conv64 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                        SwitchNorm2d(32),
                        nn.ReLU()
                       )

                                     
        self.conv643 = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1),
                                     nn.Tanh())        

        #self.up3 = nn.UpsamplingNearest2d(scale_factor=2)


        self.convvv4 = nn.Sequential(nn.Conv2d(64, 32, 5, 1, 2),
                        SwitchNorm2d(32),
                        nn.ReLU())
                       
        # Residual blocks
        res_blocks = []
        for _ in range(1):
            res_blocks.append(ResidualBlockme(32))
        self.res_blocks1 = nn.Sequential(*res_blocks)

        self.att32 = Self_Attn(32, 'relu')
        
        self.convvv5 = nn.Sequential(nn.Conv2d(32, 12, 3, 1, 1),
                        SwitchNorm2d(12),
                        nn.ReLU()
                       )

                                     
        self.convvv6 = nn.Sequential(nn.Conv2d(12, 3, 3, 1, 1),
                                     nn.Tanh())

        # heatmap detection
        self.fan = FAN()
        
        self.convfea128 = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 2),
                        SwitchNorm2d(32),
                        nn.ReLU())


    def forward(self, x, num_modules):
        
        x = self.convvv1(x)
        x = self.up1(x)
   
        xa = self.convvv2(x)
        
        x32 = self.conv32(xa)
        x32out = self.conv323(x32)
        
        x =  self.up2(xa)
        
        x64 = self.conv64(x)
        x64out = self.conv643(x64)

        x2 = self.res_blocks(x) 
        
        x3, attention1 = self.att64(x2)

        x4 = self.up3(x3)
        
        

        x5 = self.convvv4(x4)
        
        x6 = self.res_blocks1(x5)
        
        #genfeature1, attention2 =  self.att32(x6)
        
        outt = self.convvv5(x6)
        
        out = self.convvv6(outt)
        
        genfeature = self.convfea128 (out)

        ### fake images
        
        heatmap=self.fan(out)

        gen_fan=torch.cat(heatmap,0)
        
        predict_heat=gen_fan.cpu()
        
        predict_heat = predict_heat.detach().numpy()


        preheat=get_peak_points(predict_heat)
        preheat1 = preheat[0]
        prelandmark1=landmarks_68_to_5(preheat1)
        prelandmark1 =prelandmark1.astype(np.uint8)

        imgs_numpy0 = out.cpu().squeeze(0).permute(1,2,0)
        imgs_numpy = imgs_numpy0.detach().numpy()
        imgs_numpy = (imgs_numpy * 0.5 + 0.5) * 255
        imgs_numpy = imgs_numpy.astype(np.uint8)
        img_genunet = Image.fromarray(imgs_numpy)
        img_genunet = img_genunet.resize((128,128) , Image.LANCZOS)
        #print(img_genunet.width)
        batch = process(img_genunet, prelandmark1 )
        to_tensor = transforms.ToTensor()
        for k in batch:
            batch[k] = to_tensor( batch[k] )
            batch[k] = batch[k] * 2.0 - 1.0

        batch['left_eye'] = batch['left_eye'].unsqueeze(0).cuda()
        #print(batch['left_eye'].shape)
        batch['nose'] = batch['nose'].unsqueeze(0).cuda()
        #print(batch['nose'].shape)
        batch['mouth'] = batch['mouth'].unsqueeze(0).cuda()
        batch['right_eye'] = batch['right_eye'].unsqueeze(0).cuda()
        #print(out1.shape)
        out=out.cuda()
        
        gen_fan=gen_fan.cuda()
        return x32out, x64out, out, genfeature, gen_fan, batch['left_eye'], batch['nose'], batch['mouth'], batch['right_eye']   #3*128*128




class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.convv1 = nn.Sequential(nn.Conv2d(3, 32, 5, 1, 2),
                      nn.MaxPool2d(2),
                      nn.ReLU(),
                      nn.Dropout(p=0.2)
                     )

        self.convv2 = nn.Sequential(nn.Conv2d(32, 64, 5, 1, 2),
                      nn.MaxPool2d(2),
                      nn.ReLU(),
                      nn.Dropout(p=0.2)
                     )

        self.convv3 = nn.Sequential(nn.Conv2d(64, 128, 5, 1, 2),
                      nn.MaxPool2d(2),
                      nn.ReLU(),
                      nn.Dropout(p=0.2)
                     )
        self.convv4 = nn.Sequential(nn.Conv2d(128, 96, 5, 1, 2),
                      nn.ReLU(),
                      nn.MaxPool2d(2),
                      nn.Dropout(p=0.2)
                     )
        ## .reshape(8*8*96)
        # Output layer

        self.out =  nn.Sequential(
            nn.Linear(8*8*96, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024,4)
            )

        # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.convv1(img)
        x = self.convv2(x)
        x = self.convv3(x)
        x = self.convv4(x)
        #outattention2, p2 = self.attn2(outmodel)   #128*64*64
        #x = x.view(-1, 1*96*8*8)
        x = x.view(-1, 8*8*96)
        outdis = self.out(x)
        out = self.sigmoid(outdis)
        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
        #for name, module in self.submodule.items():
            if name is "avgpool":
                self.avgpool =nn.AdaptiveAvgPool2d((1,1))
            if name in self.extracted_layers:
                outputs.append(x)
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
        return outputs




class LocalPathway(nn.Module):
    def __init__(self,use_batchnorm = True,feature_layer_dim = 64 , fm_mult = 1.0):
        super(LocalPathway,self).__init__()
        n_fm_encoder = [64,128,256,512]
        n_fm_decoder = [256,128]
        n_fm_encoder = emci(n_fm_encoder,fm_mult)
        n_fm_decoder = emci(n_fm_decoder,fm_mult)
        #encoder
        self.conv0 = sequential( conv( 3   , n_fm_encoder[0]  , 3 , 1 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                ResidualBlock(n_fm_encoder[0] , activation = nn.LeakyReLU() ) )
        self.conv1 = sequential( conv( n_fm_encoder[0]  , n_fm_encoder[1] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                ResidualBlock(n_fm_encoder[1] , activation = nn.LeakyReLU() ) )
        self.conv2 = sequential( conv( n_fm_encoder[1] , n_fm_encoder[2] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                ResidualBlock(n_fm_encoder[2] , activation = nn.LeakyReLU() ) )
        self.conv3 = sequential( conv( n_fm_encoder[2] , n_fm_encoder[3] , 3 , 2 , 1 , "kaiming" , nn.LeakyReLU(1e-2) , use_batchnorm) ,
                                ResidualBlock(n_fm_encoder[3] , activation = nn.LeakyReLU() ) )
        #decoder
        self.deconv0 =   deconv( n_fm_encoder[3] , n_fm_decoder[0] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.after_select0 =  sequential(   conv( n_fm_decoder[0] + self.conv2.out_channels , n_fm_decoder[0] , 3 , 1 , 1 , 'kaiming' ,  nn.LeakyReLU() , use_batchnorm  ) ,    ResidualBlock( n_fm_decoder[0] , activation = nn.LeakyReLU()  )  )

        self.deconv1 =   deconv( self.after_select0.out_channels , n_fm_decoder[1] , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.after_select1 = sequential(    conv( n_fm_decoder[1] + self.conv1.out_channels , n_fm_decoder[1] , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm  ) ,   ResidualBlock( n_fm_decoder[1] , activation = nn.LeakyReLU()  )  )

        self.deconv2 =   deconv( self.after_select1.out_channels , feature_layer_dim , 3 , 2 , 1 , 1 , "kaiming" , nn.ReLU() , use_batchnorm)
        self.after_select2 = sequential( conv( feature_layer_dim + self.conv0.out_channels , feature_layer_dim  , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm  ) ,   ResidualBlock( feature_layer_dim , activation = nn.LeakyReLU()  )  )
        self.local_img = conv( feature_layer_dim  , 3 , 1 , 1 , 0 , None , None , False )


    def forward(self,x):
        conv0 = self.conv0( x )
        conv1 = self.conv1( conv0 )
        conv2 = self.conv2( conv1 )
        conv3 = self.conv3( conv2 )
        deconv0 = self.deconv0( conv3 )
        after_select0 = self.after_select0( torch.cat([deconv0,conv2],  1) )
        deconv1 = self.deconv1( after_select0 )
        after_select1 = self.after_select1( torch.cat([deconv1,conv1] , 1) )
        deconv2 = self.deconv2( after_select1 )
        after_select2 = self.after_select2( torch.cat([deconv2,conv0],  1 ) )
        local_img = self.local_img( after_select2 )
        assert local_img.shape == x.shape  ,  "{} {}".format(local_img.shape , x.shape)
        return  local_img , deconv2



class LocalFuser(nn.Module):
    #differs from original code here
    #https://github.com/HRLTY/TP-GAN/blob/master/TP_GAN-Mar6FS.py
    '''
    x         y
    39.4799 40.2799
    85.9613 38.7062
    63.6415 63.6473
    45.6705 89.9648
    83.9000 88.6898
    this is the mean locaiton of 5 landmarks
    '''
    def __init__(self ):
        super(LocalFuser,self).__init__()
    def forward( self , f_left_eye , f_right_eye , f_nose , f_mouth):
        EYE_W , EYE_H = 40 , 40
        NOSE_W , NOSE_H = 40 , 32
        MOUTH_W , MOUTH_H = 48 , 32
        IMG_SIZE = 128
        f_left_eye = torch.nn.functional.pad(f_left_eye , (39 - EYE_W//2  - 1 ,IMG_SIZE - (39 + EYE_W//2 - 1) ,40 - EYE_H//2 - 1, IMG_SIZE - (40 + EYE_H//2 - 1)))
        f_right_eye = torch.nn.functional.pad(f_right_eye,(86 - EYE_W//2  - 1 ,IMG_SIZE - (86 + EYE_W//2 - 1) ,39 - EYE_H//2 - 1, IMG_SIZE - (39 + EYE_H//2 - 1)))
        f_nose = torch.nn.functional.pad(f_nose,          (64 - NOSE_W//2 - 1 ,IMG_SIZE - (64 + NOSE_W//2 -1) ,64 - NOSE_H//2- 1, IMG_SIZE - (64 + NOSE_H//2- 1)))
        f_mouth = torch.nn.functional.pad(f_mouth,        (65 - MOUTH_W//2 -1 ,IMG_SIZE - (65 + MOUTH_W//2 -1),89 - MOUTH_H//2-1, IMG_SIZE - (89 + MOUTH_H//2-1)))
        return torch.max( torch.stack( [ f_left_eye , f_right_eye , f_nose , f_mouth] , dim = 0  ) , dim = 0 )[0]




class Generator(nn.Module):
    def __init__(self, use_batchnorm = True , use_residual_block = True):
        super(Generator,self).__init__()

        self.global_pathway = GeneratorResNet()
        self.local_pathway_left_eye  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_right_eye  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_nose  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_pathway_mouth  = LocalPathway(use_batchnorm = use_batchnorm)
        self.local_fuser    = LocalFuser()


        self.reconstruct_128 = sequential( *[ResidualBlock( 32+3+3+64 , activation = nn.LeakyReLU())] )
        self.conv5 = sequential( conv( self.reconstruct_128.out_channels , 48, 5 , 1 , 2 , 'kaiming' , nn.LeakyReLU() , use_batchnorm  ) , \
                ResidualBlock(48, kernel_size = 3 , activation = nn.LeakyReLU() ))
        self.conv6 = conv( 48, 32 , 3 , 1 , 1 , 'kaiming' , nn.LeakyReLU() , use_batchnorm )
        self.decoded_img128 = conv( 32 , 3 , 3 , 1 , 1 , None , activation = None )


    def forward( self, I16, use_dropout ):

        #pass through local pathway
        out32, out64, gen_unet, gen_feature, predict_heat, left_eye, nose, mouth, right_eye = self.global_pathway(I16, num_modules=1)
        le_fake , le_fake_feature = self.local_pathway_left_eye(left_eye)
        re_fake , re_fake_feature = self.local_pathway_right_eye(right_eye)
        nose_fake , nose_fake_feature = self.local_pathway_nose(nose)
        mouth_fake , mouth_fake_feature = self.local_pathway_mouth(mouth)

        #fusion
        local_feature = self.local_fuser( le_fake_feature , re_fake_feature , nose_fake_feature , mouth_fake_feature )
        local_vision = self.local_fuser( le_fake , re_fake , nose_fake , mouth_fake )
        local_input = self.local_fuser( left_eye , right_eye , nose , mouth )

        return gen_unet, out32, out64, local_vision , local_input, le_fake , re_fake , nose_fake , mouth_fake, predict_heat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)

