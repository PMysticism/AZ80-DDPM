#Cast-forged AZ80 Magnesium Alloy Microstructure Image Generation

#Select process parameters from valid enteries for each parameter below

#Process Parameter Selction

#Cast geometry {valid enteries: "cylinder" or "preform"}
cast_geometry = "cylinder"
#Metallography sample location {valid enteries: "tall", "web", or "short"}
metallography_location = "web"
#Casting cooling rate {valid enteries: "1.5", "6", or "10.4"}
casting_cooling_rate = "6"
#Soaking process {valid enteries: "normal", "1.5h", or "2h"}
soaking_process = "normal"
#Heat treatment {valid enteries: "none" or "homogenization"}
heat_treatment = "none"
#Forging temperature {valid enteries: "250", "300", or "350"}
forging_temperature = "350"
#Image magnification {valid enteries: "100", "500", "1000", "1500", "2000", or "3000"}
magnification = "1000"

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from numpy.random import randn
import torchvision.utils
from torch.distributions import uniform
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
n_sampled_images = 1

image_shape = (1,512,512)
image_size = 512
image_dim = int(np.prod(image_shape))
learning_rate = 3e-4

shapes = 2
locations = 3
cooling_rates = 3
soaking_times = 3
forging_temps = 3
heat_treatments = 2
magnifications = 6
embedding_dim = 100
num_classes = 114

shp_dict =  {'cylinder':0, 'preform':1}
loc_dict =  {'tall':0, 'web':1, 'short':2}
cr_dict =  {'1.5':0, '6':1, '10.4':2}
sk_dict =  {'normal':0, '1.5h':1, '2h':2}
ht_dict =  {'none':0, 'homogenization':1}
ft_dict =  {'250':0, '300':1, '350':2}
mag_dict =  {'100':0, '500':1, '1000':2, '1500':3, '2000':4, '3000':5}


reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t * 255.),
    ])
#---
#fig = plt.figure(figsize = (100,100))
def show_images(images, index, label):
    ax = fig.add_subplot(21, 1,index+1, xticks = [], yticks = [])
    plt.gca().set_title(label)
    #ax.set_title(label,fontsize = 40)
    plt.imshow(images.cpu(), cmap = 'gray')
    plt.show()
    

def show_grids(images):
 
    fig = plt.figure(figsize=(6.61, 6.61))
    #fig = plt.figure()
    #fig.set_dpi(100)
    grid = ImageGrid(fig, 111, 
                     nrows_ncols=(1,1),  
                     axes_pad=0.5
                     )
    plt.axis('off')
    j = 0
    for ax, im in zip(grid, images.cpu().view(-1,image_size,image_size)):
        ax.imshow(im, cmap = 'gray')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        #ax.title.set_text(labels_title[j])
        #ax.text(0.5, 0.5, 'synthesized \n ©Azqadan et al. 2023', transform=ax.transAxes,
        #fontsize=100, color='white', alpha=0.3,
        #ha='center', va='center', rotation=30)
        #ax.title.set_size(28)
        #fig.gca().set_title(labelt[i])
        j += 1
    


    
    plt.show()
    figname = str('Synthesize')
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)
          
    
def load_model(address):
    if torch.cuda.is_available() == True:
        checkpoint = torch.load(address)
        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint["ema_model_state"])
        optimizer.load_state_dict(checkpoint["model_optimizer"])
    else:
        checkpoint = torch.load(address, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint["ema_model_state"])
        optimizer.load_state_dict(checkpoint["model_optimizer"])
        
    

        

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=image_size):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:,None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*epsilon, epsilon
    
    def sample_timesteps(self,n):
        return torch.randint(low = 1, high = self.noise_steps, size = (n,))
    
    def sample(self, model, n, SHP, Loc, CR, SK, HT, FT, Mag, cfg_scale=0):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n)*i).long().to(device)
                predicted_noise = model(x,t, SHP, Loc, CR, SK, HT, FT, Mag)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x,t, None, None, None, None, None, None, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i>1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        #x = (x.clamp(-1,1)+1)/2
        #x = (x*255).type(torch.uint8)
        return x        
    
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0
        
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
            
    def update_average(self, old, new):
        if old is None:
            return new
        return old*self.beta + (1-self.beta)*new
    
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1
        
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        
    def forward(self,x):
        x = x.view(-1, self.channels, self.size*self.size).swapaxes(1,2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2,1).view(-1, self.channels, self.size, self.size)
    
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, residual = False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels,kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, imsize, emb_dim=512):
        super().__init__()
        self.imsize = imsize
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, (out_channels-7))
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
        self.SHP_label = nn.Sequential(
                                nn.Embedding(shapes, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.Loc_label = nn.Sequential(
                                nn.Embedding(locations, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.CR_label = nn.Sequential(
                                nn.Embedding(cooling_rates, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.SK_label = nn.Sequential(
                                nn.Embedding(soaking_times, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.HT_label = nn.Sequential(
                                nn.Embedding(heat_treatments, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.FT_label = nn.Sequential(
                                nn.Embedding(forging_temps, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.Mag_label = nn.Sequential(
                                nn.Embedding(magnifications, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        
    def forward(self, x, t, SHP, Loc, CR, SK, HT, FT, Mag):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        shpemb = self.SHP_label(SHP).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        locemb = self.Loc_label(Loc).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        cremb = self.CR_label(CR).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        skemb = self.SK_label(SK).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        htemb = self.HT_label(HT).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        ftemb = self.FT_label(FT).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        magemb = self.Mag_label(Mag).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        x = torch.cat((x, shpemb, locemb, cremb, skemb, htemb, ftemb, magemb), dim = 1)
        return x + emb
    
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, imsize, emb_dim=512):
        super().__init__()
        self.imsize = imsize
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels-7, in_channels // 2)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
        self.SHP_label = nn.Sequential(
                                nn.Embedding(shapes, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.Loc_label = nn.Sequential(
                                nn.Embedding(locations, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.CR_label = nn.Sequential(
                                nn.Embedding(cooling_rates, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.SK_label = nn.Sequential(
                                nn.Embedding(soaking_times, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.HT_label = nn.Sequential(
                                nn.Embedding(heat_treatments, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.FT_label = nn.Sequential(
                                nn.Embedding(forging_temps, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        self.Mag_label = nn.Sequential(
                                nn.Embedding(magnifications, embedding_dim),
                                nn.Linear(embedding_dim, 1*self.imsize*self.imsize))
        
    def forward(self, x, skip_x, t, SHP, Loc, CR, SK, HT, FT, Mag):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim = 1)
        x = self.conv(x)
        shpemb = self.SHP_label(SHP).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        locemb = self.Loc_label(Loc).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        cremb = self.CR_label(CR).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        skemb = self.SK_label(SK).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        htemb = self.HT_label(HT).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        ftemb = self.FT_label(FT).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        magemb = self.Mag_label(Mag).view(x.shape[0],1,x.shape[-2],x.shape[-1])
        x = torch.cat((x, shpemb, locemb, cremb, skemb, htemb, ftemb, magemb), dim = 1)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        return x + emb
    
    

class UNet_conditional(nn.Module):
    def __init__(self,c_in = 1, c_out = 1, time_dim=512, num_classes=None):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16,32, 256)
        self.sa1 = SelfAttention(32, 256)
        self.down2 = Down(32, 64, 128)
        self.sa2 = SelfAttention(64, 128)
        self.down3 = Down(64, 128, 64)
        self.sa3 = SelfAttention(128, 64)
        self.down4 = Down(128, 256, 32)
        self.sa4 = SelfAttention(256, 32)
        self.down5 = Down(256, 512, 16)
        self.sa5 = SelfAttention(512, 16)
        self.down6 = Down(512, 512, 8)
        self.sa6 = SelfAttention(512, 8)
        
        self.bot1 = DoubleConv(512, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 512)
        
        self.up6 = Up(1024, 256, 16)
        self.as6 = SelfAttention(256, 16)
        self.up5 = Up(512, 128, 32)
        self.as5 = SelfAttention(128, 32)
        self.up4 = Up(256, 64, 64)
        self.as4 = SelfAttention(64, 64)
        self.up3 = Up(128, 32, 128)
        self.as3 = SelfAttention(32, 128)
        self.up2 = Up(64, 16, 256)
        self.as2 = SelfAttention(16, 256)
        self.up1 = Up(32, 8, 512)
        self.as1 = SelfAttention(8, 512)
        self.outc = nn.Conv2d(8, c_out, kernel_size = 1)
        
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float().to(device) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2)*inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2)*inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t, SHP, Loc, CR, SK, HT, FT, Mag):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
            
        
        x0 = self.inc(x)
        x1 = self.down1(x0, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x1 = self.sa1(x1)
        x2 = self.down2(x1, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x2 = self.sa2(x2)
        x3 = self.down3(x2, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x3 = self.sa3(x3)
        x4 = self.down4(x3, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x4 = self.sa4(x4)
        x5 = self.down5(x4, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x5 = self.sa5(x5)
        x6 = self.down6(x5, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x6 = self.sa6(x6)
        
        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)
        
        x = self.up6(x6, x5, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x = self.as6(x)
        x = self.up5(x, x4, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x = self.as5(x)
        x = self.up4(x, x3, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x = self.as4(x)
        x = self.up3(x, x2, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x = self.as3(x)
        x = self.up2(x, x1, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x = self.as2(x)
        x = self.up1(x, x0, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x = self.as1(x)
        output = self.outc(x)
        
        
        return output
    
model = UNet_conditional(num_classes = num_classes).to(device)
ema_model = UNet_conditional(num_classes = num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
diffusion = Diffusion(img_size = image_size)


load_dir = '/kaggle/input/checkpoint2/All_CDDM_HR_Cat_6.pth.tar'
load_model(load_dir)


with torch.no_grad():
    t_shp = torch.tensor([shp_dict[cast_geometry]])
    t_loc = torch.tensor([loc_dict[metallography_location]])
    t_cr = torch.tensor([cr_dict[casting_cooling_rate]])
    t_sk = torch.tensor([sk_dict[soaking_process]])
    t_ht = torch.tensor([ht_dict[heat_treatment]])
    t_ft = torch.tensor([ft_dict[forging_temperature]])
    t_mag = torch.tensor([mag_dict[magnification]])

    t_shp = t_shp.long().to(device)
    t_loc = t_loc.long().to(device)
    t_cr = t_cr.long().to(device)
    t_sk = t_sk.long().to(device)
    t_ht = t_ht.long().to(device)
    t_ft = t_ft.long().to(device)
    t_mag = t_mag.long().to(device)
    #sampled_images = diffusion.sample(model, n_sampled_images, t_shp, t_loc, t_cr, t_sk, t_ht, t_ft, t_mag, cfg_scale=0)
    #sampled_images = reverse_transforms(sampled_images)
    ema_sampled_images = diffusion.sample(ema_model, n_sampled_images, t_shp, t_loc, t_cr, t_sk, t_ht, t_ft, t_mag, cfg_scale=0)
    ema_sampled_images = reverse_transforms(ema_sampled_images)
    #show_grids(sampled_images, test_labels,  e, label_dict)
    show_grids(ema_sampled_images)
  
