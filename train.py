import torch
import torch.utils.data as Data
import glob
import random
import os
import numpy as np
import sys
import cv2
import dlib
import torchvision


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
from math import floor
from torch.nn.init import xavier_normal , kaiming_normal
import torchvision.utils as vutils
import time
import warnings
import copy


from model.layers import *
import model.StackedHourGlass as M
from model.network import *
from model.FAN import *
from torchvision import transforms
from model.utils import resize,elementwise_mult_cast_int
#~ from spectral import SpectralNorm


torch.manual_seed(1)    # reproducible
emci = elementwise_mult_cast_int


import visdom
#~ #指定Environment：train
vis = visdom.Visdom(env='train')
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--imsize", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=10, help="interval betwen image samples")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="save model per checkpoint")
parser.add_argument("--imsize", type=int, default=128, help="image resize")
opt = parser.parse_args()
print(opt)


### define transforms
lr_transforms = [transforms.Resize((imsize//8, imsize//8)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]

hr_transforms = [transforms.Resize((imsize, imsize)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
                
hr_transforms32 = [transforms.Resize((imsize//4, imsize//4)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
                
hr_transforms64 = [transforms.Resize((imsize//2, imsize//2)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]         


### define dataset
class ImageDataset(Dataset):
    def __init__(self, root1, root2, lr_transforms=None, hr_transforms=None, hr_transforms2=None):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        self.hr_transform32 = transforms.Compose(hr_transforms32)
        self.hr_transform64 = transforms.Compose(hr_transforms64)
        
        self.files1 = sorted(glob.glob(root1 + '/*.*'))
        self.files2 = sorted(glob.glob(root2 + '/*.*'))

    def __getitem__(self, index):       
        
        img1 = Image.open(self.files1[(index) % len(self.files1)])
        img_lr1 = self.lr_transform(img1)
        img_hr1 = self.hr_transform(img1) 
        
        img2 = Image.open(self.files2[(index) % len(self.files2)])
        img_lr2 = self.lr_transform(img2)
        img_hr2 = self.hr_transform(img2) 
        img_hr32 = self.hr_transform32(img2)
        img_hr64 = self.hr_transform64(img2) 
        return {'lr1': img_lr1, 'lr2': img_lr2, 'hr1': img_hr1, 'hr2': img_hr2, 'hr32': img_hr32, 'hr64': img_hr64}

        # {'lr': img_lr, 'hr': img_hr, 'feat_points', [5x2]}
               
    def __len__(self):
        return len(self.files1)


### define dataloader
loader1 = Data.DataLoader(ImageDataset("../../data/multi_pie_bmp/nonfrontal","../../data/multi_pie_bmp/normalfrontal",
                                      lr_transforms=lr_transforms, 
                                      hr_transforms=hr_transforms),
                                      batch_size=checkpoint_interval, shuffle=True, num_workers=1)


### define perceptual loss
extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]  
res = models.resnet50(pretrained=True)
res.avgpool =nn.AdaptiveAvgPool2d((1,1))
res.cuda()
extract_result = FeatureExtractor(res, extract_list)  
    

### Initialize SRGAN Network
cuda = True if torch.cuda.is_available() else False


### Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(imsize / 2**4), int(imsize / 2**4)
patch = (patch_h, patch_w)


###  Initialize generator and discriminator
discriminator = Discriminator()
Gend = Generator()


### Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    discriminator = discriminator.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
    Gend = Gend.cuda()


### generator._initialize_weights()
discriminator._initialize_weights()
Gend._initialize_weights()


### Optimizers
optimizer_G = torch.optim.RMSprop(Gend.parameters(), lr=1e-4, alpha=0.01, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=1e-5, alpha=0.01, eps=1e-08, weight_decay=0, momentum=0, centered=False)


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_lr1 = Tensor(opt.batch_size, opt.channels, 16, 16)

input_hr1 = Tensor(opt.batch_size, opt.channels, 128, 128)
input_lr2 = Tensor(opt.batch_size, opt.channels, 16, 16)
input_hr2 = Tensor(opt.batch_size, opt.channels, 128, 128)
input_hr32 = Tensor(opt.batch_size, opt.channels, 32, 32)
input_hr64 = Tensor(opt.batch_size, opt.channels, 64, 64)

imgs_hr_tmp=Tensor(498, opt.channels,, 128, 128)


### Adversarial ground truths
valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)

### dlib 68 landmark predict
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 


def get_facial_landmarks(img):  
    rects = detector(img, 0) 
    #print(rects.shape)  
    rect = rects[0]
    #~ print(len(rects))
    #~ print("22222")
    shape = predictor(img, rect)
    a=np.array([[pt.x, pt.y] for pt in shape.parts()])    
    b=a.astype('float')   #.reshape(-1,136)
    return b 
    
    
def _putGaussianMap(center, visible_flag, crop_size_y, crop_size_x, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        grid_y = crop_size_y
        grid_x = crop_size_x
        if visible_flag == False:
            return np.zeros((grid_y,grid_x))
        #start = stride / 2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx 
        yy = yy 
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap
        
        
def _putGaussianMaps(keypoints,crop_size_y, crop_size_x, sigma):
        """

        :param keypoints: (15,2)
        :param crop_size_y: int
        :param crop_size_x: int
        :param stride: int
        :param sigma: float
        :return:
        """
        all_keypoints = keypoints
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        for k in range(point_num):
            flag = ~np.isnan(all_keypoints[k,0])
            heatmap = _putGaussianMap(all_keypoints[k],flag,crop_size_y,crop_size_x,sigma)
            heatmap = heatmap[np.newaxis,...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(heatmaps_this_img,axis=0) # (num_joint,crop_size_y/stride,crop_size_x/stride)
        return heatmaps_this_img



def adjust_learning_rate(optimizer, epoch, lrr):
       ##Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lrr * (0.99 ** (epoch // 1))
        for param_group in optimizer.param_groups:
           param_group['lr'] = lr


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



five_pts_idx = [ [36,41] , [42,47] , [27,35] , [48,48] , [54,54] ]
def landmarks_68_to_5(x):
    y = []
    for j in range(5):
        y.append( np.mean( x[five_pts_idx[j][0]:five_pts_idx[j][1] + 1] , axis = 0  ) )
    return np.array( y , np.float32)






for epoch in range(0, 400):
    m = max(1e-1*0.99**epoch, 0.05)
    adjust_learning_rate(optimizer_G,epoch,lrr=1e-4)
    
    adjust_learning_rate(optimizer_D,epoch,lrr=1e-5)
   
    print("Current epoch : {}".format(epoch))
    
    for i, imgs in enumerate(loader1):
                
        imgs_lr = Variable(input_lr1.copy_(imgs['lr1']))

        imgs_hr1 = Variable(input_hr1.copy_(imgs['hr1']))

        imgs_lr2 = Variable(input_lr2.copy_(imgs['lr2']))

        imgs_hr2 = Variable(input_hr2.copy_(imgs['hr2']))
    
        imgs_hr32 = Variable(input_hr32.copy_(imgs['hr32']))
        imgs_hr64 = Variable(input_hr64.copy_(imgs['hr64']))
     
        up2 = nn.Upsample(scale_factor=8, mode='bilinear')        

        imgs_lrup=up2(imgs_lr)                              
              
        try:   
            # Train Generators
            # ------------------
            optimizer_G.zero_grad()
            
            # Generate a high resolution image from low resolution input

            gen_unet, out32, out64, local_vision , local_input, le_fake , re_fake , nose_fake , mouth_fake , predict_heat = Gend (imgs_lr, use_dropout = True)
            
            predict_heat = predict_heat.squeeze(0)     

            imgs_numpy = imgs_hr2.cpu().squeeze(0).permute(1,2,0)
          
            imgs_numpy = imgs_numpy.numpy()
            imgs_numpy = (imgs_numpy * 0.5 + 0.5) * 255
            imgs_numpy = imgs_numpy.astype(np.uint8)                     
            real_land = get_facial_landmarks(imgs_numpy)    ## to tensor
            preland=landmarks_68_to_5(real_land)
            preland =preland.astype(np.uint8)
 
            real_heatmaps = _putGaussianMaps(real_land, 128, 128, 5)
            real_heatmaps=torch.from_numpy(real_heatmaps)
            real_heatmaps=real_heatmaps.float()
            real_heatmaps=real_heatmaps.cuda()
            
            
            real_imgs = imgs_hr2.cpu().squeeze(0).permute(1,2,0)
            real_imgs = real_imgs.detach().numpy()
            real_imgs = (real_imgs * 0.5 + 0.5) * 255
            real_imgs = real_imgs.astype(np.uint8)
            real_imgs = Image.fromarray(real_imgs)
            real_imgs = real_imgs.resize((128,128) , Image.LANCZOS)
            batchGT = process( real_imgs, preland )
            to_tensor = transforms.ToTensor()
            for k in batchGT:
                batchGT[k] = to_tensor( batchGT[k] )
                batchGT[k] = batchGT[k] * 2.0 - 1.0
            
            batchGT['left_eye'] = batchGT['left_eye'].unsqueeze(0).cuda()
            batchGT['right_eye'] = batchGT['right_eye'].unsqueeze(0).cuda()
            batchGT['nose'] = batchGT['nose'].unsqueeze(0).cuda()
            batchGT['mouth'] = batchGT['mouth'].unsqueeze(0).cuda()

           
            # FACIAL PART loss
            eyel_loss = criterion_content( le_fake , batchGT['left_eye'] )
            eyer_loss = criterion_content( re_fake , batchGT['right_eye'] )
            nose_loss = criterion_content( nose_fake , batchGT['nose'] )
            mouth_loss = criterion_content( mouth_fake , batchGT['mouth'] )
            loss_pixel_local = eyel_loss + eyer_loss + nose_loss + mouth_loss
            
            
            # Adversarial loss
            gen_validity = discriminator(gen_unet)
          
            loss_GAN = criterion_GAN(gen_validity, valid)
            
            
            # Resnet50 loss
            real_features2 = Variable(extract_result(imgs_hr2)[4].data, requires_grad=False)   
            unet_features = extract_result(gen_unet)[4]    #128                             
                    
            loss_content2 =  criterion_content(unet_features, real_features2)    
            
            I64_fa = extract_result(out64)[4]    #128
            
            real_features3 = Variable(extract_result(imgs_hr64)[4].data, requires_grad=False)          
                    
            loss_content3 =  criterion_content(I64_fa, real_features3)                  
            
            lossid = loss_content2 + loss_content3
            
            
            # l2 loss

            loss_G1= criterion_GAN(gen_unet, imgs_hr2) 
            
            loss_G32= criterion_GAN(out32, imgs_hr32) 
            
            loss_G64= criterion_GAN(out64, imgs_hr64) 
            
            lossl2 = loss_G1 + loss_G32 + loss_G64

            # landmark loss
                                            
            loss_land = criterion_GAN(predict_heat, real_heatmaps)/68
            #print(loss_land)

            # Total loss                        
            loss_G = lossl2  +  (1e-1) * loss_GAN +  (1e-1) * lossid + loss_land + 3 * loss_pixel_local
                    
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator
            # ------------------
            optimizer_D.zero_grad()
            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr2), valid)
            loss_fake = criterion_GAN(discriminator(gen_unet.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
           
            curr_progress = "[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [G loss: {:f}][D loss: {:f}][Pixel loss: {:f}]".format(epoch, 400, i, len(loader1), loss_G.item(), loss_D.item(), loss_pixel_local.item())
            print(curr_progress)
            
            # save log
            with open("my_log.txt", 'a') as f:
                f.write(curr_progress + "\n")
    
    
            batches_done = epoch * len(loader1) + i
            if batches_done % 100 == 0:
                # Save image sample
                save_image(torch.cat((gen_unet.data, imgs_hr2.data), -2),
                            'images/%d.png' % batches_done, normalize=True)
                vis.line(X=torch.FloatTensor([batches_done]), Y=torch.FloatTensor([loss_G]), win='trainG', update='append' if epoch > 0 else None,
                    opts={'title': 'G loss'})
                vis.line(X=torch.FloatTensor([batches_done]), Y=torch.FloatTensor([loss_D]), win='trainD', update='append' if epoch > 0 else None,
                    opts={'title': 'D loss'})
    
    
            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
               torch.save(Gend.state_dict(), 'saved_models/Gend_%d.pth' % epoch)
               torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
        except:
               #print("i")   
               number = "[Epoch {:d}/{:d}] [Batch {:d}/{:d}] ]".format(epoch, 400, i, len(loader1))
            
               # save log
               with open("difficult image.txt", 'a') as f:
                    f.write(number + "\n")
                          
        
            
    
