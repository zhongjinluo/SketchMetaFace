from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur


class TrainDataset(Dataset):

    def __init__(self):
        # Path setup
        self.root = "../datasets/"
        self.DEPTH_PREV = os.path.join(self.root, 'HDn')
        self.DEPTH_NEXT = os.path.join(self.root, 'HD0')
        self.FLOW = os.path.join(self.root, 'FLOWs')
        self.MASK = os.path.join(self.root, 'MASK0')
        self.subjects = self.get_subjects()
        self.augs = 1

        self.to_tensor_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        self.to_tensor_flow = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5), (0.5, 0.5))
        ])


    def get_img_info(self, subject):
        render_list = []
    
        depth_prev_path = os.path.join(self.DEPTH_PREV, subject+'.png')
        depth_prev = cv2.imread(depth_prev_path, -1)
        depth_prev = np.array(depth_prev/255.0/255.0).reshape(512, 512, 1)
        depth_prev = self.to_tensor_d(depth_prev).float()

        depth_next_path = os.path.join(self.DEPTH_NEXT, subject+'.png')
        depth_next = cv2.imread(depth_next_path, -1)
        depth_next = np.array(depth_next/255.0/255.0).reshape(512, 512, 1)
        depth_next = self.to_tensor_d(depth_next).float()

        # print(depth_next.shape, depth_prev.shape)

        dd = torch.cat([depth_prev, depth_next])

        flow_pth = os.path.join(self.FLOW, subject+'.npz')
        flow = np.load(flow_pth, allow_pickle=True)['flow']
        flow = self.to_tensor_flow(flow).float()

        mask_path = os.path.join(self.MASK, subject+'.png')
        mask = cv2.imread(mask_path, -1)
        mask = torch.from_numpy(mask).float()

        return {
            'DD': dd,
            "FLOW": flow,
            "MASK0": mask
        }

    def get_subjects(self):
        all_subjects = []
        files = os.listdir(self.FLOW)
        for file in files:
            all_subjects.append(file[:-4])
        return all_subjects

    def __len__(self):
        return len(self.subjects) * self.augs


    def get_item(self, index):
        sid = index // self.augs
        subject = self.subjects[sid]
        data = self.get_img_info(subject)
        return data

    def __getitem__(self, index):
        return self.get_item(index)