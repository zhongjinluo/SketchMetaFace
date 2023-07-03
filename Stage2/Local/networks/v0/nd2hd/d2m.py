from PIL import Image
import numpy as np
import torch
import os
import cv2


projection_matrix = np.array([[0.8333333, 0., 0., 0.], [ 0., 0.8333333, 0., -0.], [0., 0., -0.1, 0.], [ 0., 0., 0., 1.]])
model_view_matrix = np.array([[1, 0., 0., 0.], [ 0., 1, 0., -0.], [0., 0., 1, -3.6], [ 0., 0., 0., 1.]])
pmat_trans_inv = np.linalg.inv(np.transpose(projection_matrix))
vmat_trans_inv = np.linalg.inv(np.transpose(model_view_matrix))
pmat_trans_inv = pmat_trans_inv[0:3, 0:3]
vmat_trans_inv = vmat_trans_inv[0:3, 0:4]

def depth2mesh(data_uint16, save_path, name_uint16, image_size=512):

    data_uint16 = cv2.resize(data_uint16, (image_size,image_size))
    data_uint16 = np.flip(data_uint16, 1)
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

    vert = uv_mat
    # vert[:, 0] = vert[:, 0]
    # vert[:, 1] = vert[:, 1]
    vert[:, 2] = dep_map[vid, uid]/255./255.
    vert[:, 2] = vert[:, 2] * 2 - 1
    
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
    
    return vid, uid, vert

for f in os.listdir("training_dataset/outputs/"):
    img = cv2.imread("training_dataset/outputs/"+f, -1)
    depth2mesh(img, "/data3/xiaojin/Characters_PIFU/SimpModeling_Colored/demos/mesh/", f[0:-4], image_size=512)
