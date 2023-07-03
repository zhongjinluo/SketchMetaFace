import json
import torch
import os, sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.append("./networks/v0/SDFRenderer")
from core.utils.render_utils import *
from core.utils.decoder_utils import load_decoder
from core.visualize.visualizer import *
from core.visualize.vis_utils import *
from core.sdfrenderer import SDFRenderer_color as SDFRenderer
from common.geometry import *

from networks.v0.sketch2model import Sketch2Model

class MyRenderer:
    def __init__(self, s2m):
        img_h, img_w = 137, 137
        intrinsic = np.array([[150., 0., 68.5], [0., 150., 68.5], [0., 0., 1.]])
        load_size = 512
        self.to_tensor = transforms.Compose([
            transforms.Resize(load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        output_shape = 224
        img_hw = (img_h, img_w)
        resize = self.resize_intrinsic(intrinsic, output_shape / img_h)
        self.sdf_renderer = SDFRenderer(s2m.netG, None, resize, march_step=100, buffer_size=1, threshold=5e-4, ray_marching_ratio=1.0)
        self.camera_list = self.generate_camera_list(self.sdf_renderer)
    
    def resize_intrinsic(self, intrinsic, scale):
        intrinsic[:2] = intrinsic[:2] * scale
        return intrinsic

    def generate_camera_list(self, sdf_renderer):
        K = sdf_renderer.get_intrinsic()
        camera_list = []
        for d in [-90, 0, 180]:
            view = View(d, 0, 0, 2.5)
            RT = view.get_extrinsic()
            camera_list.append(Camera(K, RT))
        return camera_list

    def render(self, s2m, norm, f=""):
        norm = self.to_tensor(Image.fromarray(norm).convert('RGB'))
        render = torch.cat([norm], 0).unsqueeze(0)
        data = {}
        with torch.no_grad():
            data['img'] = render
            data['name'] = "current"
            data['b_min'] = np.array([-1, -1, -1])
            data['b_max'] = np.array([1, 1, 1])
            data['calib'] = s2m.calib.unsqueeze(0)
        image_tensor = data['img'].cuda()
        self.sdf_renderer.decoder.filter(image_tensor)
        names = ["Front", "Left", "Right"]
        for idx, camera in enumerate(self.camera_list):
            demo_color_save_render_output("/data1/zhongjin/Characters_PIFU/SimpModeling_Colored/networks/v0/renders/"+f+names[idx], self.sdf_renderer, None, None, camera)
        return True

if __name__ == '__main__':
    s2m = Sketch2Model()
    r = MyRenderer(s2m)
    norm_dir = "/data1/zhongjin/Characters_PIFU/pifuhd_sdf_depth/DIST-Renderer/renders/DeepSketch2Character8_NORMAL_PIX2PIX/"
    for f in os.listdir(norm_dir):
        norm = np.array(Image.open(os.path.join(norm_dir, f)).convert('RGB'))
        print(f)
        r.render(s2m, norm)
        break