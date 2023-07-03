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
        self.root = "../data/"
        self.NORM = os.path.join(self.root, 'N')
        self.CDEPTH = os.path.join(self.root, 'RHD0_camera')
        self.RDEPTH = os.path.join(self.root, 'RHDn_camera')
        print(self.CDEPTH, self.RDEPTH)
        self.subjects = self.get_subjects()
        self.augs = 1

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

        self.to_tensor2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])


    def get_img_info(self, subject):
        render_list = []
        fnorm_path = os.path.join(self.NORM, subject.replace("_a_1", "")+'.png')
        fnorm = Image.open(fnorm_path).convert('RGB')
        fnorm = fnorm.resize((512, 512))
        fnorm = self.to_tensor(fnorm).float()

        cdepth_path = os.path.join(self.CDEPTH, subject+'.png')
        cdepth = cv2.imread(cdepth_path, -1)
        cdepth = np.array(cdepth/255.0/255.0).reshape(512, 512, 1)
        cdepth = self.to_tensor2(cdepth).float()

        ncd = torch.cat([fnorm, cdepth])

        rdepth_path = os.path.join(self.RDEPTH, subject+'.png')
        rdepth = cv2.imread(rdepth_path, -1)
        rdepth = np.array(rdepth/255.0/255.0).reshape(512, 512, 1)
        rdepth = self.to_tensor2(rdepth).float()
        return {
            'NCD': ncd,
            "RD": rdepth
        }

    def get_subjects(self):
        all_subjects = []
        files = os.listdir(self.CDEPTH)
        for file in files:
            if "_a_1.png" in file:
            	all_subjects.append(file[:-4])
        print("dataset size: ", len(all_subjects))
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
