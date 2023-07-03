import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# 1 modified
from networks.v0.pix2pixHD.options.test_options import TestOptions
from networks.v0.pix2pixHD.models.models import create_model
import networks.v0.pix2pixHD.util.util as util
from networks.v0.pix2pixHD.util import html

class Sketch2Norm:
    def __init__(self):
        self.opt = TestOptions().parse(save=False)
        # 2 modified
        # self.opt.which_epoch = 40
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.name = "GapMesh"
        
        self.opt.which_epoch = 70
        self.opt.checkpoints_dir = "./checkpoints/pix2pixHD_12_all/"

        print("Load Sketch2Norm:", self.opt.checkpoints_dir)

        self.opt.label_nc = 0
        self.opt.dataroot = ""
        
        # 4 modified
        self.opt.loadSize = 512
        self.opt.no_instance = True
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.model = create_model(self.opt).cuda()


    def predict(self, image):
        with torch.no_grad():
            input_img = self.to_tensor(image).unsqueeze(0).cuda()
            generated = self.model.inference(input_img, None, None)
            norm = util.tensor2im(generated.data[0])
            return image, norm

if __name__ == '__main__':
    pass