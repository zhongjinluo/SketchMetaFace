import sys
import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# 1 modified
from networks.v0.pifu_sdf_dit_human.lib.options import BaseOptions
from networks.v0.pifu_sdf_dit_human.lib.mesh_util import *
from networks.v0.pifu_sdf_dit_human.lib.sample_util import *
from networks.v0.pifu_sdf_dit_human.lib.train_util import *
from networks.v0.pifu_sdf_dit_human.lib.model import *

# from pifuhd_sdf.lib.options import BaseOptions
# from pifuhd_sdf.lib.mesh_util import *
# from pifuhd_sdf.lib.sample_util import *
# from pifuhd_sdf.lib.train_util import *
# from pifuhd_sdf.lib.model import *



class Sketch2Model:
    def __init__(self, projection_mode='orthogonal'):
        opt = BaseOptions().parse()
        opt.resolution = 256
        # 2 modified
        opt.loadSize = 512
        self.opt = opt

        self.load_size = opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        netG = HGPIFuNetwNML(opt, projection_mode).cuda()
        netG.is_train = False

        # 3 modified
        # netG.load_state_dict(torch.load(os.path.join("networks/v0/pifu_sdf_dit_human/checkpoints/example", "netG_latest"), map_location=cuda))
        netG.load_state_dict(torch.load(os.path.join("checkpoints/pifu_sdf_DeepImplicitTemplates", "netG_latest"), map_location=cuda))

        # # # 4 matrix
        # M = np.load('./templates/matrix.npz')
        # self.M = {}
        # W = M["W"]
        # self.M["smooth_weight"] = W[0].item()
        # self.M["laplacian_weight"] = W[1].item()
        # self.M["target_weight"] = W[2].item()
        # self.M["A"] = torch.from_numpy(M["A"]).float().cuda()
        # R_INV = torch.from_numpy(M["R_INV"]).float().cuda()
        # Q_T = torch.from_numpy(M["Q_T"]).float().cuda()
        # self.M["QR"] = R_INV @ Q_T
        # self.M["V"] = torch.from_numpy(M["V"]).float()
        # self.M["F"] = M["F"]
        # self.M["L"] = torch.from_numpy(M["L"]).float().cuda()
        # self.M["D"] = torch.zeros((self.M["L"].shape[0], 3)).cuda()


        self.cuda = cuda
        self.netG = netG

        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "PV.json"), 'r') as f:
            data = json.load(f)
            P = np.array(data["P"]).reshape(-1, 4)
            P[1, 1] = -P[1, 1]
            V = np.array(data["V"]).reshape(-1, 4)
            self.calib = torch.from_numpy(P.dot(V)).float()
            self.netG.calib_tensor = self.calib.cuda()

    def predict(self, norm, save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "m.obj")):
        norm = self.to_tensor(Image.fromarray(norm).convert('RGB'))
        render = torch.cat([norm], 0).unsqueeze(0)
        data = {}
        with torch.no_grad():
            self.netG.eval()
            data['img'] = render
            data['name'] = "current"
            data['b_min'] = np.array([-1, -1, -1])
            data['b_max'] = np.array([1, 1, 1])
            data['calib'] = self.calib.unsqueeze(0)
            return gen_mesh_server(self.opt, self.netG, self.cuda, data, save_path, use_octree=False)
    
    def predict_with_template(self, norm, save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "m.obj")):
        norm = self.to_tensor(Image.fromarray(norm).convert('RGB'))
        render = torch.cat([norm], 0).unsqueeze(0)
        data = {}
        with torch.no_grad():
            self.netG.eval()
            data['img'] = render
            data['name'] = "current"
            data['b_min'] = np.array([-1, -1, -1])
            data['b_max'] = np.array([1, 1, 1])
            data['calib'] = self.calib.unsqueeze(0)
            return gen_mesh_with_template(self.opt, self.netG, self.cuda, data, save_path, M=self.M)

    def predict_with_template_2(self, norm, save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "m.obj"), landmarks=[]):
        norm = self.to_tensor(Image.fromarray(norm).convert('RGB'))
        render = torch.cat([norm], 0).unsqueeze(0)
        data = {}
        with torch.no_grad():
            self.netG.eval()
            data['img'] = render
            data['name'] = "current"
            data['b_min'] = np.array([-1, -1, -1])
            data['b_max'] = np.array([1, 1, 1])
            data['calib'] = self.calib.unsqueeze(0)
            return gen_mesh_with_template_2(self.opt, self.netG, self.cuda, data, save_path, M=self.M, landmarks=landmarks)

if __name__ == '__main__':
    s2m = Sketch2Model()
    sketch = np.array(Image.open("/data1/zhongjin/Characters_PIFU/SimpModeling_Colored/networks/v0/test_reduced/TOPO_FL_SC_mesh_Zooey+Deschanel_C00015.png").convert('RGB'))
    from sketch2norm import Sketch2Norm
    s2n = Sketch2Norm()
    s, n = s2n.predict(sketch)
    Image.fromarray(n).save("n.png")
    # s2m.predict(n)
    s2m.predict_with_template_2(n)