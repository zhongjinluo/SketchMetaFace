import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# 1 modified
from options.test_options import TestOptions
from models.models import create_model
import util.util as util
from util import html


class Sketch2Depth:
    def __init__(self):
        self.opt = TestOptions().parse(save=False)
        # 2 modified
        # self.opt.which_epoch = 40
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.name = "GapMesh"
        self.opt.input_nc = 6 
        
        self.opt.which_epoch = "latest"
        self.opt.checkpoints_dir = "./checkpoints/"
        
        print("Load Sketch2Depth:", self.opt.checkpoints_dir)

        self.opt.label_nc = 0
        self.opt.dataroot = ""
        
        # 4 modified
        self.opt.loadSize = 512
        self.opt.no_instance = True
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.model = create_model(self.opt).cuda()

            # transform_A = get_transform(self.opt, params)
            # A_tensor = transform_A(A.convert('RGB'))
            # A2_tensor = transform_A(A2.convert('RGB'))
            # A_tensor = torch.cat([A_tensor, A2_tensor])

    def predict(self, S, D):
        with torch.no_grad():
            A_tensor = self.to_tensor(S)
            A2_tensor = self.to_tensor(D)
            A_tensor = torch.cat([A_tensor, A2_tensor]).unsqueeze(0).cuda()
            generated = self.model.inference(A_tensor, None, None)
            refined_depth = util.tensor2im(generated.data[0])
            return S, refined_depth

if __name__ == '__main__':
    s2n = Sketch2Depth()

    '''
    root = "/program/SIGRAPH22/pix2pix-es/pix2pixHD_2022_N/datasets/faces/test_A/"
    for f in os.listdir(root):
        S = np.array(Image.open("/program/SIGRAPH22/pix2pix-es/pix2pixHD_2022_N/datasets/faces/test_A/" + f).convert('RGB'))
        D = np.array(Image.open("/program/SIGRAPH22/pix2pix-es/pix2pixHD_2022_N/datasets/faces/test_A2/" + f).convert('RGB'))
        _, d = s2n.predict(S, D)
        d = Image.fromarray(d)
        d.save("outputs/" + f)
        print(f)
    '''
    root = "/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/sketch/"
    for f in os.listdir(root):
        S = np.array(Image.open("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/sketch/" + f).convert('RGB'))
        D = np.array(Image.open("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/depth/" + f).convert('RGB'))
        _, n = s2n.predict(S, D)
        n = Image.fromarray(n)
        n.save("outputs_wild/" + f)
        print(f)