import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


# 1 modified
from networks.v0.pix2pixHD_2022.options.test_options import TestOptions
from networks.v0.pix2pixHD_2022.models.models import create_model
import networks.v0.pix2pixHD_2022.util.util as util
from networks.v0.pix2pixHD_2022.util import html

# from pix2pixHD_2022.options.test_options import TestOptions
# from pix2pixHD_2022.models.models import create_model
# import pix2pixHD_2022.util.util as util
# from pix2pixHD_2022.util import html


class Sketch2Norm:
    def __init__(self):
        self.opt = TestOptions().parse(save=False)
        # 2 modified
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.name = "GapMesh_low"
        self.opt.input_nc = 6 
        
        self.opt.which_epoch = "latest"
        self.opt.checkpoints_dir = "./checkpoints/pix2pixHD_2022_N/checkpoints/"
        
        print("Load Sketch2Norm:", self.opt.checkpoints_dir)

        self.opt.label_nc = 0
        self.opt.dataroot = ""
        
        # 4 modified
        self.opt.loadSize = 512
        self.opt.no_instance = True
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.model = create_model(self.opt).cuda()

    def predict(self, S, D):
        with torch.no_grad():
            A_tensor = self.to_tensor(S)
            A2_tensor = self.to_tensor(D)
            A_tensor = torch.cat([A_tensor, A2_tensor]).unsqueeze(0).cuda()
            generated = self.model.inference(A_tensor, None, None)
            norm = util.tensor2im(generated.data[0])
            return S, norm

if __name__ == '__main__':
    s2n = Sketch2Norm()
    # root = "/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/sketch/"
    # for f in os.listdir(root):
    #     S = np.array(Image.open("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/sketch/" + f).convert('RGB'))
    #     D = np.array(Image.open("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/depth/" + f).convert('RGB'))
    #     _, n = s2n.predict(S, D)
    #     n = Image.fromarray(n)
    #     n.save(f)
    #     print(f)

    root = "/program/SIGRAPH22/ALGO_V7/test_norm/sketch/"
    for f in os.listdir(root):
        S = np.array(Image.open("/program/SIGRAPH22/ALGO_V7/test_norm/sketch/" + f).convert('RGB'))
        D = np.array(Image.open("/program/SIGRAPH22/ALGO_V7/test_norm/depth/" + f).convert('RGB'))
        _, n = s2n.predict(S, D)
        n = Image.fromarray(n)
        n.save("/program/SIGRAPH22/ALGO_V7/test_norm/" + f)
        print(f)