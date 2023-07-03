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
from networks.v0.pifu_sdf_DeepImplicitTemplates_Depth.lib.options import BaseOptions
from networks.v0.pifu_sdf_DeepImplicitTemplates_Depth.lib.mesh_util import *
from networks.v0.pifu_sdf_DeepImplicitTemplates_Depth.lib.sample_util import *
from networks.v0.pifu_sdf_DeepImplicitTemplates_Depth.lib.train_util import *
from networks.v0.pifu_sdf_DeepImplicitTemplates_Depth.lib.model import *



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

        netG.load_state_dict(torch.load(os.path.join("checkpoints/pifu_sdf_DeepImplicitTemplates_Depth/checkpoints/example", "netG_latest"), map_location=cuda))



        self.cuda = cuda
        self.netG = netG

        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "PV.json"), 'r') as f:
            data = json.load(f)
            P = np.array(data["P"]).reshape(-1, 4)
            P[1, 1] = -P[1, 1]
            V = np.array(data["V"]).reshape(-1, 4)
            self.calib = torch.from_numpy(P.dot(V)).float()
            self.netG.calib_tensor = self.calib.cuda()

    def predict(self, norm, depth, save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "m.obj")):
        norm = self.to_tensor(Image.fromarray(norm).convert('RGB'))
        depth = self.to_tensor(Image.fromarray(depth).convert('RGB'))
        render = torch.cat([norm, depth], 0).unsqueeze(0)
        data = {}
        with torch.no_grad():
            self.netG.eval()
            data['img'] = render
            data['name'] = "current"
            data['b_min'] = np.array([-1, -1, -1])
            data['b_max'] = np.array([1, 1, 1])
            data['calib'] = self.calib.unsqueeze(0)
            return gen_mesh_server(self.opt, self.netG, self.cuda, data, save_path, use_octree=False)
    
    def predict_part(self, norm, depth, vertices_list, faces_list, save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "m.obj")):
        norm = self.to_tensor(Image.fromarray(norm).convert('RGB'))
        depth = self.to_tensor(Image.fromarray(depth).convert('RGB'))
        render = torch.cat([norm, depth], 0).unsqueeze(0)
        data = {}
        with torch.no_grad():
            self.netG.eval()
            data['img'] = render
            data['name'] = "current"
            data['b_min'] = np.array([-1, -1, -1])
            data['b_max'] = np.array([1, 1, 1])
            data['calib'] = self.calib.unsqueeze(0)
            return gen_mesh_part(self.opt, self.netG, self.cuda, data, vertices_list, faces_list, save_path, use_octree=False)
    
    def predict_sub_mc(self, norm, depth, vertices_list, faces_list, save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "m.obj")):
        norm = self.to_tensor(Image.fromarray(norm).convert('RGB'))
        depth = self.to_tensor(Image.fromarray(depth).convert('RGB'))
        render = torch.cat([norm, depth], 0).unsqueeze(0)
        data = {}
        with torch.no_grad():
            self.netG.eval()
            data['img'] = render
            data['name'] = "current"
            data['b_min'] = np.array([-1, -1, -1])
            data['b_max'] = np.array([1, 1, 1])
            data['calib'] = self.calib.unsqueeze(0)
            return gen_mesh_sub_mc(self.opt, self.netG, self.cuda, data, vertices_list, faces_list, save_path, use_octree=False)

def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices, center, r
    
if __name__ == '__main__':
    import random
    s2m = Sketch2Model()
    '''
    root = "/program/SIGRAPH22/pix2pix-es/pix2pixHD_2022_D/outputs/"
    file_list = []
    for f in os.listdir(root):
        file_list.append(f)
    random.shuffle(file_list)
    for f in file_list:
        n = np.array(Image.open("/program/SIGRAPH22/pix2pix-es/pix2pixHD_2022_N/outputs/" + f).convert('RGB'))
        d = np.array(Image.open("/program/SIGRAPH22/pix2pix-es/pix2pixHD_2022_D/outputs/" + f).convert('RGB'))
        s2m.predict(n, d, "outputs/" + f.replace(".png", ".obj"))
        print(f)
        # break
    '''
    root = "/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/sketch/"
    file_list = []
    for f in os.listdir(root):
        file_list.append(f)
    random.shuffle(file_list)
    for f in file_list:
        n = np.array(Image.open("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER3/gallery/normal/" + f).convert('RGB'))
        d = np.array(Image.open("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/depth/" + f).convert('RGB'))
        s2m.predict(n, d, "outputs_wild3/" + f.replace(".png", "_raw.obj"))
        n = np.array(Image.open("/program/SIGRAPH22/pix2pix-es/pix2pixHD_2022_N/outputs_wild/" + f).convert('RGB'))
        d = np.array(Image.open("/program/SIGRAPH22/pix2pix-es/pix2pixHD_2022_D/outputs_wild/" + f).convert('RGB'))
        d0 = np.array(Image.open("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/depth/" + f).convert('RGB'))
        r0 = np.mean(d0) / 255
        r = np.mean(d) / 255
        bias = (r - r0) * 8.666
        s2m.predict(n, d, "outputs_wild3/" + f.replace(".png", "_mc.obj"))
        coarse = trimesh.load("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER2/gallery/coarse/" + f.replace(".png", ".obj"), process=False)
        coarse.export("outputs_wild3/" + f.replace(".png", "_coarse0.obj"))
        coarse.vertices[:, 2] += bias
        coarse.export("outputs_wild3/" + f.replace(".png", "_coarse.obj"))
        vertices_list_xiaojin, faces_list_xiaojin = s2m.predict_sub_mc(n, d, [[], coarse.vertices, []], [[], coarse.faces, []], "outputs_wild3/" + f.replace(".png", "_deform.obj"))
        # break