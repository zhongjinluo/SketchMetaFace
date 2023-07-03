import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import cv2

from util_model import ResnetBlock, DecoderBlock, EncoderBlock, get_norm_layer 


class AE_encoder(nn.Module):
    def __init__(self,latent_dim=512,input_nc=1):
        super(AE_encoder,self).__init__()
        image_size = 128
        latent_dim = latent_dim #512
        #print(latent_dim)
        # norm_layer iamge_size input_nc latent_dim
        latent_size = int(image_size/16)
        longsize = 256 * latent_size * latent_size
        #print(longsize,latent_size)

        activation = nn.ReLU()
        padding_type = 'reflect'
        norm_layer=nn.BatchNorm2d

        layers_list = []
        # encode
        layers_list.append(EncoderBlock(channel_in=input_nc, channel_out=32, kernel_size=4, padding=1, stride=2))  # 8 32 64 64 
        
        dim_size = 32
        for i in range(3):
            layers_list.append(ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer)) 
            layers_list.append(EncoderBlock(channel_in=dim_size, channel_out=dim_size*2, kernel_size=4, padding=1, stride=2)) 
            dim_size *= 2
        
        layers_list.append(ResnetBlock(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  
        
        # final shape Bx256*7*6
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))#,
        #print("longsize",longsize)
        # self.fc_var = nn.Sequential(nn.Linear(in_features=longsize, out_features=latent_dim))#,
    def forward(self,x):
        x = self.conv(x)
        #print(x.shape)
        x = torch.reshape(x,[x.size()[0],-1])
        #print(x.shape)
        mu = self.fc_mu(x)
        return mu

class AE_decoder(nn.Module):
    def __init__(self, latent_dim=512,output_nc=1):  
        super(AE_decoder, self).__init__()
        image_size = 128
        #print(latent_dim)
        # norm_layer, image_size, output_nc ,latent_dim
        latent_size = int(image_size/16)
        self.latent_size = latent_size
        longsize = 256*latent_size*latent_size

        activation = nn.ReLU()
        padding_type='reflect'
        norm_layer=nn.BatchNorm2d

        self.fc = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=longsize))
        layers_list = []

        layers_list.append(ResnetBlock(256, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 
         
        dim_size = 128
        for i in range(3):
            layers_list.append(DecoderBlock(channel_in=dim_size*2, channel_out=dim_size, kernel_size=4, padding=1, stride=2, output_padding=0)) #latent*2
            layers_list.append(ResnetBlock(dim_size, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  
            dim_size = int(dim_size/2)
        
        layers_list.append(DecoderBlock(channel_in=32, channel_out=32, kernel_size=4, padding=1, stride=2, output_padding=0)) #352 352
        layers_list.append(ResnetBlock(32, padding_type=padding_type, activation=activation, norm_layer=norm_layer))  # 176 176 

        # layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #96*160
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(32,output_nc,kernel_size=5,padding=0))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, x):
        # print("in DecoderGenerator, print some shape ")
        # print(ten.size())
        x = self.fc(x)
        # print(ten.size())
        #import pdb;pdb.set_trace()
        x = torch.reshape(x,(x.size()[0],256, self.latent_size, self.latent_size))
        # print(ten.size())
        
        x = self.conv(x)

        return x


class AE_model(nn.Module):
    def __init__(self,latent_dim=512,norm='instance',input_nc=3,output_nc=3):
        #init all layers
        super(AE_model,self).__init__()
        self.encoder = AE_encoder(latent_dim=latent_dim,input_nc=input_nc)
        self.decoder = AE_decoder(latent_dim=latent_dim,output_nc=output_nc)
    def forward(self,x):
        # x is the image after  process
        #calculate the result
        latent = self.encoder(x)
        recover =  self.decoder(latent)
        return recover


class sqare_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        #y is label
        empty = (y.sum(1)==3) # B*size*size
        pixel_loss = torch.pow((x-y),2).sum(1)
        return torch.mean(pixel_loss + pixel_loss*empty) # add more loss on empty place
