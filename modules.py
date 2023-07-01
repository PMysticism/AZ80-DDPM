import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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
        

class SelfAttention2(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention2, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
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
    

class SelfAttention4(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention4, self).__init__()
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
    
    
class UNet(nn.Module):
    def __init__(self,c_in = 1, c_out = 1, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16,32)
        self.sa1 = SelfAttention4(32, 128)
        self.down2 = Down(32, 64)
        self.sa2 = SelfAttention4(64, 64)
        self.down3 = Down(64, 128)
        self.sa3 = SelfAttention4(128, 32)
        self.down4 = Down(128, 256)
        self.sa4 = SelfAttention4(256, 16)
        self.down5 = Down(256, 256)
        self.sa5 = SelfAttention4(256, 8)
        
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        
        self.up5 = Up(512, 128)
        self.as5 = SelfAttention4(128, 16)
        self.up4 = Up(256, 64)
        self.as4 = SelfAttention4(64, 32)
        self.up3 = Up(128, 32)
        self.as3 = SelfAttention4(32, 64)
        self.up2 = Up(64, 16)
        self.as2 = SelfAttention4(16, 128)
        self.up1 = Up(32, 8)
        self.as1 = SelfAttention4(8, 256)
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
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        x0 = self.inc(x)
        x1 = self.down1(x0, t)
        #x1 = self.sa1(x1)
        x2 = self.down2(x1, t)
        x2 = self.sa2(x2)
        x3 = self.down3(x2, t)
        x3 = self.sa3(x3)
        x4 = self.down4(x3, t)
        x4 = self.sa4(x4)
        x5 = self.down5(x4, t)
        x5 = self.sa5(x5)
        
        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)
        
        
        x = self.up5(x5, x4, t)
        #del x5, x4
        x = self.as5(x)
        x = self.up4(x, x3, t)
        #del x3
        x = self.as4(x)
        x = self.up3(x, x2, t)
        #del x2
        x = self.as3(x)
        x = self.up2(x, x1, t)
        #del x1
        #x = self.as2(x)
        x = self.up1(x, x0, t)
        #del x0
        #x = self.as1(x)
        x = self.outc(x)
        return x
    

class UNet_conditional(nn.Module):
    def __init__(self,c_in = 1, c_out = 1, time_dim=512, num_classes=None):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16,32, 256)
        self.sa1 = SelfAttention4(32, 256)
        self.down2 = Down(32, 64, 128)
        self.sa2 = SelfAttention4(64, 128)
        self.down3 = Down(64, 128, 64)
        self.sa3 = SelfAttention2(128, 64)
        self.down4 = Down(128, 256, 32)
        self.sa4 = SelfAttention4(256, 32)
        self.down5 = Down(256, 512, 16)
        self.sa5 = SelfAttention4(512, 16)
        self.down6 = Down(512, 512, 8)
        self.sa6 = SelfAttention4(512, 8)
        
        self.bot1 = DoubleConv(512, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 512)
        
        self.up6 = Up(1024, 256, 16)
        self.as6 = SelfAttention4(256, 16)
        self.up5 = Up(512, 128, 32)
        self.as5 = SelfAttention4(128, 32)
        self.up4 = Up(256, 64, 64)
        self.as4 = SelfAttention4(64, 64)
        self.up3 = Up(128, 32, 128)
        self.as3 = SelfAttention2(32, 128)
        self.up2 = Up(64, 16, 256)
        self.as2 = SelfAttention4(16, 256)
        self.up1 = Up(32, 8, 512)
        self.as1 = SelfAttention4(8, 512)
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
        x = self.as3(x)
        x = self.up2(x, x1, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x = self.as2(x)
        x = self.up1(x, x0, t, SHP, Loc, CR, SK, HT, FT, Mag)
        #x = self.as1(x)
        output = self.outc(x)
        
        
        return output
