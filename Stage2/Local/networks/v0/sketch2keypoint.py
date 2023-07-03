import os
import numpy as np
import cv2
import time
import sys
import torch
import numpy as np
import openmesh as om
from networks.v0.sdfrenderer import MyRenderer
sys.path.append("./networks/v0/pfld/Front")
from model2 import MobileNetV2, MyResNest50
from networks.v0.sketch2model import Sketch2Model
from PIL import Image

class Sketch2Keypoint:
    def __init__(self):
        # self.front = MyResNest50(nums_class=45)
        self.front = torch.load("/data1/zhongjin/Characters_PIFU/detector/224_depth/Front/checkpoints/model_950.pth").cuda()
        self.front.eval()

        # self.left = MyResNest50(nums_class=24)
        self.left = torch.load("/data1/zhongjin/Characters_PIFU/detector/224_depth/Left/checkpoints/model_750.pth").cuda()
        self.left.eval()

        self.right = torch.load("/data1/zhongjin/Characters_PIFU/detector/224_depth/Right/checkpoints/model_800.pth").cuda()
        self.right.eval()

        self.model_dict = {
            "Front": self.front,
            "Left": self.left,
            "Right": self.right
        }

    def predict(self, norm, depth):

        landmarks = []
        for k, m in self.model_dict.items():
            normal = cv2.imread("/data1/zhongjin/Characters_PIFU/SimpModeling_Colored/networks/v0/renders/%s_normal.png" % k)
            depth = cv2.imread("/data1/zhongjin/Characters_PIFU/SimpModeling_Colored/networks/v0/renders/%s_depth.png" % k)
            normal = normal.copy()
            normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
            normal = normal.astype(np.float32) / 256.0
            depth = depth.copy()
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            depth = depth.astype(np.float32) / 256.0
            images = np.concatenate([normal, depth], axis=2)
            images = np.expand_dims(images, 0)
            images = torch.Tensor(images.transpose((0, 3, 1, 2)))
            pre_landmarks, _ = m(images.cuda())
            pre_landmark = pre_landmarks[0].cpu().detach().numpy()

            land = pre_landmark.reshape(-1, 3)
            landmarks.append(land)
        landmarks = np.concatenate(landmarks, axis=0)
        mesh = om.PolyMesh(points=landmarks)
        om.write_mesh("/data1/zhongjin/Characters_PIFU/SimpModeling_Colored/networks/v0/renders/landmark.obj", mesh)
        return landmarks

if __name__ == '__main__':
    s2m = Sketch2Model()
    s2k = Sketch2Keypoint()
    r = MyRenderer(s2m)
    norm_dir = "/data1/zhongjin/Characters_PIFU/pifuhd_sdf_depth/DIST-Renderer/renders/DeepSketch2Character8_NORMAL_PIX2PIX/"
    for f in os.listdir(norm_dir):
        print(f)
        if "TOPO" not in f:
            continue
        norm = np.array(Image.open(os.path.join(norm_dir, f)).convert('RGB'))
        r.render(s2m, norm)
        break
    s2k.predict("", "")
