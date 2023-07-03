# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet_hd import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters_hd import HGFilter
from ..net_util import init_net
from .SurfaceClassifier import SurfaceClassifier
#from ..networks import define_G
import cv2

class HGPIFuNetwNML(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, 
                 opt, 
                 projection_mode='orthogonal',
                 criteria={'occ': nn.L1Loss()}
                 ):
        super(HGPIFuNetwNML, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'hg_pifu'

        filter_in_ch = 6
        self.opt = opt
        self.image_filter = HGFilter(4, 2, filter_in_ch, 256, 
                                     'batch', 'ave_pool', False)

        self.mlp = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Tanh())

        self.spatial_enc = DepthNormalizer(opt)

        self.num_views=1
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.phi = None
        self.nmltrain = False

        self.intermediate_preds_list = []

        init_net(self)

        self.netF = None
        self.netB = None

        self.nmlF = None
        self.nmlB = None

    def loadFromHGHPIFu(self, net):
        hgnet = net.image_filter
        pretrained_dict = hgnet.state_dict()            
        model_dict = self.image_filter.state_dict()

        pretrained_dict = {k: v for k, v in hgnet.state_dict().items() if k in model_dict}                    

        for k, v in pretrained_dict.items():                      
            if v.size() == model_dict[k].size():
                model_dict[k] = v

        not_initialized = set()
               
        for k, v in model_dict.items():
            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                not_initialized.add(k.split('.')[0])
        
        print('not initialized', sorted(not_initialized))
        self.image_filter.load_state_dict(model_dict) 

        pretrained_dict = net.mlp.state_dict()            
        model_dict = self.mlp.state_dict()

        pretrained_dict = {k: v for k, v in net.mlp.state_dict().items() if k in model_dict}                    

        for k, v in pretrained_dict.items():                      
            if v.size() == model_dict[k].size():
                model_dict[k] = v

        not_initialized = set()
               
        for k, v in model_dict.items():
            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                not_initialized.add(k.split('.')[0])
        
        print('not initialized', sorted(not_initialized))
        self.mlp.load_state_dict(model_dict) 

    def filter(self, images):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''
        if self.nmltrain:
            
            nmls = []
        # if you wish to train jointly, remove detach etc.
            with torch.no_grad():
                if self.netF is not None:
                    self.nmlF = self.netF.forward(images).detach()
                    nmls.append(self.nmlF)
                if self.netB is not None:
                    self.nmlB = self.netB.forward(images).detach()
                    nmls.append(self.nmlB)
            if len(nmls) != 0:
                nmls = torch.cat(nmls,1)
                if images.size()[2:] != nmls.size()[2:]:
                    nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
                images = torch.cat([images,nmls],1)


        self.im_feat_list, tempx, self.normx = self.image_filter(images)

        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        
    def query(self, points, calibs, transforms=None, labels=None, update_pred=True, update_phi=True):
        '''
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            labels: [B, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''
        #print('=== calibs: ',calibs)
        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        #print('=== xyz: ',xyz)

        # if the point is outside bounding box, return outside.
        # in_bb = (xyz >= -1) & (xyz <= 1)
        # in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :]
        # in_bb = in_bb[:, None, :].detach().float()

        # if labels is not None:
        #     self.labels = in_bb * labels
        self.labels = labels

        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        intermediate_preds_list = []

        phi = None
        flag = True
        for i, im_feat in enumerate(self.im_feat_list):
            '''
            if flag:
                img = im_feat[0][1].detach().cpu().numpy()
                out = img.reshape(128, 128, 1)
                cv2.imwrite("im_feat.png", out*255)
                check = np.ones((128, 128, 1))
                XY = (xy[0] + 1) * 64
                XY = XY.detach().cpu().numpy().astype(int)
                pixels = self.index(im_feat, xy)[0][0].detach().cpu().numpy()
                for i in range(len(pixels)):
                    check[XY[1, i], XY[0, i]] = pixels[i]
                cv2.imwrite("check.png", check*255)
                check2 = np.ones((128, 128, 1))
                check2[XY[1], XY[0]] = 0.0
                cv2.imwrite("check2.png", check2*255)
                flag = False
            '''
            point_local_feat_list = [self.index(im_feat, xy), sp_feat]       
            point_local_feat = torch.cat(point_local_feat_list, 1)
            #pred, phi = self.mlp(point_local_feat)
            pred = self.mlp(point_local_feat)
            # print(torch.min(pred), torch.max(pred))
            # pred = in_bb * pred
            # print("in bb", torch.min(pred), torch.max(pred))

            intermediate_preds_list.append(pred)
        
        if update_phi:
            self.phi = phi

        if update_pred:
            self.intermediate_preds_list = intermediate_preds_list
            self.preds = self.intermediate_preds_list[-1]

    def calc_normal(self, points, calibs, transforms=None, labels=None, delta=0.01, fd_type='forward'):
        '''
        return surface normal in 'model' space.
        it computes normal only in the last stack.
        note that the current implementation use forward difference.
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            delta: perturbation for finite difference
            fd_type: finite difference type (forward/backward/central) 
        '''
        pdx = points.clone()
        pdx[:,0,:] += delta
        pdy = points.clone()
        pdy[:,1,:] += delta
        pdz = points.clone()
        pdz[:,2,:] += delta

        if labels is not None:
            self.labels_nml = labels

        points_all = torch.stack([points, pdx, pdy, pdz], 3)
        points_all = points_all.view(*points.size()[:2],-1)
        xyz = self.projection(points_all, calibs, transforms)
        xy = xyz[:, :2, :]
        # xy[:, 1, :] = -xy[:, 1, :]

        im_feat = self.im_feat_list[-1]
        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        point_local_feat_list = [self.index(im_feat, xy), sp_feat]            
        point_local_feat = torch.cat(point_local_feat_list, 1)

        pred = self.mlp(point_local_feat)[0]

        pred = pred.view(*pred.size()[:2],-1,4) # (B, 1, N, 4)

        # divide by delta is omitted since it's normalized anyway
        dfdx = pred[:,:,:,1] - pred[:,:,:,0]
        dfdy = pred[:,:,:,2] - pred[:,:,:,0]
        dfdz = pred[:,:,:,3] - pred[:,:,:,0]

        nml = -torch.cat([dfdx,dfdy,dfdz], 1)
        nml = F.normalize(nml, dim=1, eps=1e-8)

        self.nmls = nml

    def get_im_feat(self):
        '''
        return the image filter in the last stack
        return:
            [B, C, H, W]
        '''
        return self.im_feat_list[-1]


    def get_error(self, gamma):
        '''
        return the loss given the ground truth labels and prediction
        '''
        if self.nmltrain:
            error = {}
            error['Err(occ)'] = 0
            for preds in self.intermediate_preds_list:
                error['Err(occ)'] += self.criteria['occ'](preds, self.labels, gamma)
        
            error['Err(occ)'] /= len(self.intermediate_preds_list)
        
            if self.nmls is not None and self.labels_nml is not None:
                error['Err(nml)'] = self.criteria['nml'](self.nmls, self.labels_nml)
        else:
            error = 0
            #print('=== labels: ', self.labels.shape)
            for preds in self.intermediate_preds_list:
                #print('=== preds:', preds.shape)
                # print(preds.shape, self.labels.shape)
                preds = torch.clamp(preds, -0.1, 0.1)
                labels = torch.clamp(self.labels, -0.1, 0.1)
                error += self.criteria['occ'](preds, labels)
            error /= len(self.intermediate_preds_list)

        return error

    def forward(self, images, points, calibs, labels, gamma='mean', points_nml=None, labels_nml=None):
        self.filter(images)
        self.query(points, calibs, labels=labels)
        if points_nml is not None and labels_nml is not None:
            self.calc_normal(points_nml, calibs, labels=labels_nml)
        res = self.get_preds()
            
        err = self.get_error(gamma)

        return res, err
