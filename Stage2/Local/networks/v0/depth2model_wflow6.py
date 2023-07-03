
import torch
import sys
import cv2
import numpy as np
import os
import trimesh
import json
from multiprocessing import Process, Queue
from PIL import Image
import openmesh as om
import torchvision.transforms as transforms
from scipy.spatial import cKDTree

sys.path.append("../local/")
from networks.v0.nd2hd.render.camera import Camera
from networks.v0.nd2hd.render.gl.color_render import ColorRender
from networks.v0.nd2hd.model import Generate

import time

from scipy.linalg import expm, norm
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

class RenderProcess(Process):
    def __init__(self, vertices, faces, queue):
        super(RenderProcess,self).__init__()
        self.cam = Camera(width=1.0, height=1.0)
        self.cam.ortho_ratio = 2.4
        self.cam.near = -10
        self.cam.far = 10
        self.cam.center = np.array([0, 0, 0])
        self.cam.eye = np.array([0, 0, 3.6])
        self.verts = vertices
        self.faces = faces
        self.q = queue
        self.Ms = [M([0, 1, 0], 0), M([0, 1, 0], np.pi / 4), M([0, 1, 0], -np.pi / 4), M([1, 0, 0], np.pi / 4), M([1, 0, 0], -np.pi / 4)]
    def run(self):
        for M0 in self.Ms:
            vertices = np.dot(M0, np.array(self.verts).T).T
            depth_vals = (vertices[:, 2] + 1) * 0.5
            depth_colors = np.zeros_like(vertices)
            for k in range(0, 3):
                depth_colors[:, k] = depth_vals
            renderer_depth = ColorRender(width=512, height=512)
            renderer_depth.set_camera(self.cam)
            renderer_depth.set_mesh(vertices, self.faces, depth_colors, self.faces)
            renderer_depth.display()
            CD = renderer_depth.get_color(0)
            CD = cv2.cvtColor(CD, cv2.COLOR_RGBA2GRAY)
            self.q.put(CD)
            break
        return

projection_matrix = np.array([[0.8333333, 0., 0., 0.], [ 0., 0.8333333, 0., -0.], [0., 0., -0.1, 0.], [ 0., 0., 0., 1.]])
model_view_matrix = np.array([[1, 0., 0., 0.], [ 0., 1, 0., -0.], [0., 0., 1, -3.6], [ 0., 0., 0., 1.]])
pmat_trans_inv = np.linalg.inv(np.transpose(projection_matrix))
vmat_trans_inv = np.linalg.inv(np.transpose(model_view_matrix))
pmat_trans_inv = pmat_trans_inv[0:3, 0:3]
vmat_trans_inv = vmat_trans_inv[0:3, 0:4]
def depth2mesh(data_uint16, save_path, name_uint16, image_size=512):
    data_uint16 = cv2.resize(data_uint16, (image_size,image_size))
    # data_uint16 = np.flip(data_uint16, 1)
    dep_map = data_uint16.copy()
    img_h, img_w = dep_map.shape
    h, w = dep_map.shape
    vid, uid = np.where(dep_map < 250*250)
    nv = len(vid)
    out_name = os.path.join(save_path,"%s_d2m"%name_uint16)
    ### calculate the inverse point cloud
    uv_mat = np.ones((nv, 3), dtype=np.float16)
    uv_mat[:, 0] = (uid - img_h/2.)/img_h*2.
    uv_mat[:, 1] = (img_h/2. - vid)/img_h*2.
    vert = np.matmul(uv_mat, np.matmul(pmat_trans_inv, vmat_trans_inv))[:, 0: 3]
    vert[:, 2] = dep_map[vid, uid]/255./255.
    vert[:, 2] = vert[:, 2] * 2 - 1
    # '''
    f = open(out_name + '.obj', 'w')
    nv = 0
    vidx_map = np.full_like(dep_map, fill_value=-1, dtype=np.int)
    for i in range(0, len(vid)):
        f.write('v %f %f %f\n' % (vert[i][0], vert[i][1], vert[i][2]))
        vidx_map[vid[i], uid[i]] = nv
        nv += 1
    for i in range(0, h-2):
        for j in range(0, w-2):
            if vidx_map[i, j] >= 0 and vidx_map[i, j+1] >= 0 and vidx_map[i+1, j] >= 0 and vidx_map[i+1, j+1] >= 0:
                f.write('f %d %d %d\n' % (vidx_map[i , j] + 1, vidx_map[i + 1, j] + 1, vidx_map[i, j + 1] + 1))
                f.write('f %d %d %d\n' % (vidx_map[i + 1, j + 1] + 1, vidx_map[i, j + 1] + 1, vidx_map[i + 1, j] + 1))
    f.close()
    # '''
    return vert

def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def get_indices(mesh_vertices, sigma=1e-5):
    indices = np.where(mesh_vertices[:, 2] > sigma)[0]
    return indices

def get_indices_plus(mesh_vertices, sigma=1e-1):
    indices = np.where(mesh_vertices[:, 2] > sigma)[0]
    return indices

def project(points, calibrations):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane/return: XY: [2, N] Tensor of XY coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    return pts

class Depth2Model:
    def __init__(self):
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        self.to_tensor2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        self.generator = Generate(4, 1)
        model_CKPT = torch.load("./networks/v0/nd2hd/checkpoints/dicts/latest.pth")
        self.generator.load_state_dict(model_CKPT)
        self.generator.cuda().eval()

        with open("PV.json", 'r') as f:
            data = json.load(f)
            P = np.array(data["P"]).reshape(-1, 4)
            P[1, 1] = -P[1, 1]
            P[2, 2] = -P[2, 2]
            V = np.array(data["V"]).reshape(-1, 4)
            self.calib = torch.from_numpy(P.dot(V)).float()

        self.Ms = [M([0, 1, 0], 0), 
            M([0, 1, 0], np.pi / 4), 
            M([0, 1, 0], -np.pi / 4), 
            M([1, 0, 0], np.pi / 4), 
            M([1, 0, 0], -np.pi / 4)]

        self.Ms_re = [M([0, 1, 0], 0), 
                        M([0, 1, 0], -np.pi / 4), 
                        M([0, 1, 0], np.pi / 4), 
                        M([1, 0, 0], -np.pi / 4), 
                        M([1, 0, 0], np.pi / 4)]

    def predict(self, N, CD, out_dir=None, name=None):
        with torch.no_grad():
            N = self.to_tensor(N).float()
            CD = self.to_tensor2(CD).float()
            NCD = torch.cat([N, CD])
            pred = self.generator(NCD.unsqueeze(0).cuda().float())
            preds = (pred.permute(0,2,3,1) + 1) / 2 * 255.0
            preds = preds.cpu().detach().numpy()
            RD = np.array(preds[0,:,:,:] * 255.0, dtype=np.uint16).reshape(512, 512)
            if out_dir:
                depth2mesh(RD, out_dir, name)
        return RD

    def predict_from_coarse_model(self, S_list, vertices, faces, out_dir=None, name=None):
        start = time.time()
        vertices = normalize(vertices)
        # render
        q = Queue()
        p = RenderProcess(vertices, faces, q)
        p.start()
        while q.qsize() < 1:
            continue
        CD_list = []
        for di in range(1):
            CD = q.get()
            CD_list.append(CD)
            CD_img = np.array(CD) * 255.0 * 255.0
            CD_img = CD_img.astype(np.uint16)
            # cv2.imwrite(os.path.join(out_dir.replace("HD_ENHANCEMENT", "HD_COARSE"), name+"_c_a_"+str(di+1)+".png"), CD_img)
        end = time.time()
        print("render:", end-start)

        # predict
        start = time.time()
        Tns = []
        with torch.no_grad():
            S_tensor_list  = []
            for s in S_list:
                S_tensor_list.append(self.to_tensor(s))
            CD_tensor_list = []
            for cd in CD_list:
                CD_tensor_list.append(self.to_tensor2(cd))
            S_tensor = torch.stack(S_tensor_list).float()
            CD_tensor = torch.stack(CD_tensor_list[0:1]).float()

            SCD = torch.cat([S_tensor, CD_tensor], axis=1)
            
            outputs = self.generator(SCD.cuda().float())
            end = time.time()
            print("depth_predict:", end-start)

            depth_preds = (outputs.permute(0,2,3,1) + 1) / 2 * 255.0
            depth_preds = depth_preds.cpu().detach().numpy()
            Tns = []
            for di in range(5):
                RD = np.array(depth_preds[di,:,:,:] * 255.0, dtype=np.uint16).reshape(512, 512)
                cv2.imwrite(os.path.join(out_dir, name+"_p_a_"+str(di+1)+".png"), RD)

                verts = depth2mesh(RD, out_dir, name+"_a_"+str(di+1))
                if di == 0:
                    V = np.dot(self.Ms_re[di], verts.T).T
                    Tns.append(V)
                else:
                    verts = verts[verts[:, 2]>0.999]
                    V = np.dot(self.Ms_re[di], verts.T).T
                    Tns.append(V)
                break
        end = time.time()
        print("depth_predict+2mesh:", end-start)

        # deformation
        T0 = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        A = trimesh.smoothing.laplacian_calculation(T0)
        start = time.time()
        for j in [1]:
            V = np.dot(self.Ms[j-1], np.array(T0.vertices).T).T
            TND = depth_preds[j-1, :, :, 0]
            if j == 1:
                front_indices = get_indices(V)
                temp_f = front_indices
                points = torch.from_numpy(V[front_indices].T).float()
                pts = project(points.unsqueeze(0), self.calib.unsqueeze(0))
                xy = pts[:, :2, :]
                XY = (xy[0] + 1) * 512 * 0.5
                XY = XY.numpy().astype(int)
                Z = TND[XY[1], XY[0]] / 255.0 / 255.0
                V[front_indices, 2] = Z * 2 - 1
                Tns.append(V)
            break
        merge_vertices = np.concatenate(Tns)
        tree = cKDTree(merge_vertices)
        k = 5
        for i in range(1):
            indices = tree.query(T0.vertices, k)[1]
            if k > 1:
                V = np.mean(merge_vertices[indices], axis=1)
            else:
                V = merge_vertices[indices]
            T0.vertices = A.dot(V)
        T0.export(out_dir + name + ".obj")
        end = time.time()
        print("depth_deformation:", end-start)
        return T0.vertices, T0.faces

if __name__ == '__main__':
    d2m = Depth2Model()
    
    root = "/program/SIGRAPH22/ALGO_V6/check_problem2/3dviewer/data/normal"
    for f in os.listdir(root):
        f = "1649602788.3640742.png"
        N_list = []
        N = Image.open(os.path.join(root, f))
        N_list.append(N)
        obj_path = os.path.join("/program/SIGRAPH22/ALGO_V6/check_problem2/3dviewer/data/M2+/", f.replace(".png", ".obj"))
        if os.path.exists(obj_path):
            CM = trimesh.load(os.path.join("/program/SIGRAPH22/ALGO_V6/check_problem2/3dviewer/data/M2+/", f.replace(".png", ".obj")), process=False)
            RD = d2m.predict_from_coarse_model(N_list, CM.vertices, CM.faces, "/program/SIGRAPH22/ALGO_V6/check_problem2/3dviewer/data/output_wflow5_plus6/", f.replace(".png", ""))
            print(f)
        break