import util_model
import cv2
import torch
from autoencoder import AE_model
import numpy as np
import os
from PIL import Image
import time
from torch.utils.data import Dataset,DataLoader

def image_process(img,binary=False):
    if binary:      
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        _, img = cv2.threshold(img, 127, 255,cv2.THRESH_BINARY)    
    if len(img.shape)==3:     
        img = torch.tensor(img).permute(2,0,1)   
    else:     
        img = torch.tensor(img).unsqueeze(0)   
    img = (img*1.0/255) *2 - 1 #-1 to 1
    return img

class ImageEmbed():
    def __init__(self):
        #model = AE_model(latent_dim=1024,input_nc=3,output_nc=3)
        model_path = "checkpoint/0109_128_all/myautencoder69.pth"
        self.model = torch.load(model_path).cuda()
        self.datapath = "data/STROKES_BLACK/"
        self.targetpath = "data/STROKES_RGB/"
        if not os.path.exists('data/datasetvec.npy'):
            self.generate_npy()
        self.vecdic = np.load('data/datasetvec.npy',allow_pickle=True).item()
        # build network
        # load weight
    def generate_npy(self):
        dic_0 = {} #level 0 dic
        index = 0
        print("generate_npy")
        for cate in os.listdir(self.datapath):
            dic_1 = {}
            dic_0[cate]=dic_1
            for part in os.listdir(os.path.join(self.datapath,cate)):
                dic_2 = {}
                dic_1[part]=dic_2
                for imgname  in os.listdir(os.path.join(self.datapath,cate,part)):
                    # print(os.path.join(self.datapath,cate,part,imgname))
                    img = cv2.imread(os.path.join(self.datapath,cate,part,imgname))    
                    embedding = self.embed(img)
                    index += 1
                    if index%20==0:
                        print(index)
                    dic_2[imgname]=embedding
        
        np.save("datasetvec.npy",dic_0)
    def embed(self,img):
        image = image_process(img,binary=True)
        image = image.unsqueeze(0).cuda()
        embedding = self.model.encoder(image)
        return embedding.detach().cpu()
    def get_nearest(self,img,dtype=['human','RIGHT_EYE'], k_n=10):
        #  input img: array
        t1 = time.time()
        vec = self.embed(img)
        vec = vec.squeeze(0).numpy()
        vecnorm = np.linalg.norm(vec)
        vec = vec/vecnorm
        
        namedic = self.vecdic[dtype[0]][dtype[1]]
        keys = list(namedic.keys())
        targetvec = []
        for k in keys:
            targetvec.append(namedic[k].reshape(1,-1))
        targetvec  = np.concatenate(targetvec)
        targetvec = targetvec/np.linalg.norm(targetvec,2,axis=1).reshape(-1,1)
        cosine =  np.dot(vec,targetvec.T).reshape(-1)
        indices = cosine.argsort()[-k_n:][::-1]
        results = [keys[index] for index in indices]
        print(indices, np.argmax(cosine))
        resultpaths = [os.path.join(self.targetpath, dtype[0], dtype[1], result) for result in results]
        return resultpaths

# embder = ImageEmbed()
# #embder.get_nearest_k("demo.png")
# img = cv2.imread("/data3/xiaojin/Stroke/Myautoencoder/STROKES_SPLIT/monkey/LEFT_EYE/Animal_01_1_猴子_153_1.png")
# print(img.shape)
# img2 = embder.get_nearest(img, ['monkey','LEFT_EYE'])
# print(img2.shape)
