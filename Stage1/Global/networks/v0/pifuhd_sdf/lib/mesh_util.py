from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure
import time
import trimesh
import json
import openmesh as om
import numpy as np
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from scipy.sparse import identity, csr_matrix

def reconstructionLM(net, cuda, calib_tensor, target,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=1000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    locX = int(target[0])
    locY = int(target[1])
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid_LM(locX, locY, resolution,
                              b_min, b_max, transform=transform)
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]

        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    print('=== target: ', target)
    print('=== coords: ', coords.shape)
    print('=== sdf: ', np.max(sdf), np.min(sdf), np.mean(sdf))
    ind = np.unravel_index(np.argmax(sdf, axis=None), sdf.shape)
    print('=== lm[0] should at: ', coords[:,ind[0], ind[1], ind[2]])
    return ind, sdf

def get_bbox(vertices):
    B_MIN = []
    B_MAX = []
    B_MIN = np.min(vertices, axis=0).reshape(-1, 3) - 0.05
    B_MAX = np.max(vertices, axis=0).reshape(-1, 3) + 0.05
    return np.concatenate([B_MIN, B_MAX], axis=0)

def reconstruction_part(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None):

    mesh =  trimesh.load("/data3/xiaojin/Characters_PIFU/TEST2/sphere.obj", process=False)
    V = torch.from_numpy(mesh.vertices).float().cuda()
    F = mesh.faces
    A = torch.from_numpy(trimesh.smoothing.laplacian_calculation(mesh).toarray()).float().cuda()
    with torch.no_grad():
        for i in range(20):
            V = get_new_vertices_part(net, cuda, calib_tensor, V, F, A=A, step = 0.8/(i+1))
            print(i)
    vertices = V.detach().cpu().numpy()
    # print("deformation:", end-start)
    return vertices, F

def get_new_vertices_part(net, cuda, calib_tensor, V_old, faces, A = None, step=0.018):
    mesh = trimesh.Trimesh(vertices=V_old.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()
    # smooth normal
    if A != None:
        vertex_normals_all = A.mm(vertex_normals_all)
    V_indices = torch.Tensor(range(0, vertex_normals_all.shape[0], 1)).long()
    V_splits = torch.chunk(V_indices, 2, dim=0)
    steps = 200
    V_new_list = []
    for split in V_splits:
        V = V_old[split] 
        vertex_normals = vertex_normals_all[split]
        bias = torch.linspace(0.0, step, steps=steps).cuda()
        bias = bias.reshape(-1, 1).repeat(1, V.shape[0])
        bias = bias.reshape(-1, V.shape[0], 1)
        net.query(V.T.unsqueeze(0), calib_tensor)
        logits = net.get_preds()[0][0]
        initial_sign = torch.sign(logits)
        directions = initial_sign.reshape(-1, 1) * vertex_normals
        directions = -directions
        V_sample =  V.unsqueeze(0).repeat(bias.shape[0], 1, 1) + bias * directions
        V_sample = V_sample.reshape(-1, 3)
        net.query(V_sample.T.unsqueeze(0), calib_tensor)
        logits = net.get_preds()[0][0]
        sign = torch.sign(logits).reshape(bias.shape[0], -1, 1)
        sign = sign - sign[0, :, :]
        sign = sign.squeeze(-1).T
        sign = torch.ne(sign, 0) + 0
        idx =  torch.arange(sign.shape[1], 0, -1).cuda()
        sign= sign * idx
        indices_2 = torch.argmax(sign, 1, keepdim=True).reshape(-1)  # torch.argmax(sign, 1, keepdim=True).reshape(-1) - 1
        indices_1 = torch.arange(0, sign.shape[0], 1).cuda()
        V_sample =  V_sample.reshape(bias.shape[0], -1, 3)
        V_new = V_sample[indices_2, indices_1, :]
        V_new_list.append(V_new)
    V_new = torch.cat(V_new_list, axis=0)
    return V_new

'''
def reconstruction_with_template(net, cuda, calib_tensor, resolution, b_min, b_max, use_octree=False, M=None, num_samples=10000, transform=None):
    if M == None:
        return
    start = time.time()
    smooth_weight = M["smooth_weight"]
    laplacian_weight = M["laplacian_weight"]
    target_weight = M["target_weight"]
    QR = M["QR"]
    V = M["V"].clone().cuda()
    F = M["F"]
    L = M["L"]
    D = M["D"]
    end = time.time()
    print("M:", end-start)
    start = time.time()
    with torch.no_grad():
        for i in range(10):
            D2 = L.mm(V)
            V_bar = get_new_vertices(net, cuda, calib_tensor, V, F, A=None, step = 0.8/(i+1))
            D_V_bar = torch.cat([D * smooth_weight, D2 * laplacian_weight, V_bar * target_weight])
            V = QR @ D_V_bar
            V = V[0:D.shape[0]]
            print(i)
    vertices = V.detach().cpu().numpy()
    end = time.time()
    print("deformation:", end-start)
    return vertices, F

def get_new_vertices(net, cuda, calib_tensor, V_old, faces, A = None, step=0.018):
    mesh = trimesh.Trimesh(vertices=V_old.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()
    # smooth normal
    if A != None:
        vertex_normals_all = A.mm(vertex_normals_all)
    V_indices = torch.Tensor(range(0, vertex_normals_all.shape[0], 1)).long()
    V_splits = torch.chunk(V_indices, 10, dim=0)
    steps = 200
    V_new_list = []
    for split in V_splits:
        V = V_old[split] 
        vertex_normals = vertex_normals_all[split]
        bias = torch.linspace(0.0, step, steps=steps).cuda()
        bias = bias.reshape(-1, 1).repeat(1, V.shape[0])
        bias = bias.reshape(-1, V.shape[0], 1)
        net.query(V.T.unsqueeze(0), calib_tensor)
        logits = net.get_preds()[0][0]
        initial_sign = torch.sign(logits)
        directions = initial_sign.reshape(-1, 1) * vertex_normals
        directions = -directions
        V_sample =  V.unsqueeze(0).repeat(bias.shape[0], 1, 1) + bias * directions
        V_sample = V_sample.reshape(-1, 3)
        net.query(V_sample.T.unsqueeze(0), calib_tensor)
        logits = net.get_preds()[0][0]
        sign = torch.sign(logits).reshape(bias.shape[0], -1, 1)
        sign = sign - sign[0, :, :]
        sign = sign.squeeze(-1).T
        sign = torch.ne(sign, 0) + 0
        idx =  torch.arange(sign.shape[1], 0, -1).cuda()
        sign= sign * idx
        indices_2 = torch.argmax(sign, 1, keepdim=True).reshape(-1)  # torch.argmax(sign, 1, keepdim=True).reshape(-1) - 1
        indices_1 = torch.arange(0, sign.shape[0], 1).cuda()
        V_sample =  V_sample.reshape(bias.shape[0], -1, 3)
        V_new = V_sample[indices_2, indices_1, :]
        V_new_list.append(V_new)
    V_new = torch.cat(V_new_list, axis=0)
    return V_new
'''


# 迭代法
def reconstruction_with_template(net, cuda, calib_tensor, resolution, b_min, b_max, use_octree=False, M=None, num_samples=10000, transform=None):
    if M == None:
        return
    start = time.time()
    V = M["V"].clone().cuda()
    F = M["F"]
    A = M["A"]
    end = time.time()
    print("M:", end-start)
    start = time.time()
    with torch.no_grad():
        for i in range(6):
            if i < 1:
                V = get_new_vertices(net, cuda, calib_tensor, V, F, A=A, flag=False, step = 0.8/(i+1))
            else:
                V = get_new_vertices(net, cuda, calib_tensor, V, F, A=A, flag=True, step = 0.8/(i*10+1))
            # mesh = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=F, process=False)
            # mesh.export(str(i)+".obj")
            print(i)
    vertices = V.detach().cpu().numpy()
    end = time.time()
    print("deformation:", end-start)
    return vertices, F


def get_new_vertices(net, cuda, calib_tensor, V_old, faces, A = None, flag=False, step=0.018):
    mesh = trimesh.Trimesh(vertices=V_old.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()
    # smooth normal
    if A != None and flag == False:
        vertex_normals_all = A.mm(vertex_normals_all)
    V_indices = torch.Tensor(range(0, vertex_normals_all.shape[0], 1)).long()
    V_splits = torch.chunk(V_indices, 1, dim=0)
    steps = 100
    V_new_list = []
    for split in V_splits:
        V = V_old[split] 
        vertex_normals = vertex_normals_all[split]
        bias = torch.linspace(0.0, step, steps=steps).cuda()
        bias = bias.reshape(-1, 1).repeat(1, V.shape[0])
        bias = bias.reshape(-1, V.shape[0], 1)
        net.query(V.T.unsqueeze(0), calib_tensor)
        logits = net.get_preds()[0][0]
        initial_sign = torch.sign(logits)
        directions = initial_sign.reshape(-1, 1) * vertex_normals
        directions = -directions
        V_sample =  V.unsqueeze(0).repeat(bias.shape[0], 1, 1) + bias * directions
        V_sample = V_sample.reshape(-1, 3)
        net.query(V_sample.T.unsqueeze(0), calib_tensor)
        logits = net.get_preds()[0][0]
        sign = torch.sign(logits).reshape(bias.shape[0], -1, 1)
        sign = sign - sign[0, :, :]
        sign = sign.squeeze(-1).T
        sign = torch.ne(sign, 0) + 0
        idx =  torch.arange(sign.shape[1], 0, -1).cuda()
        sign= sign * idx
        indices_2 = torch.argmax(sign, 1, keepdim=True).reshape(-1)  # torch.argmax(sign, 1, keepdim=True).reshape(-1) - 1
        indices_1 = torch.arange(0, sign.shape[0], 1).cuda()
        V_sample =  V_sample.reshape(bias.shape[0], -1, 3)
        V_new = V_sample[indices_2, indices_1, :]
        V_new_list.append(V_new)
    V_new = torch.cat(V_new_list, axis=0)
    if flag:
        vertex_normals_all = vertex_normals_all.detach().cpu().numpy()
        indices = np.argwhere(vertex_normals_all[:, 2]>0)
        mask = np.ones(vertex_normals_all.shape[0], dtype=bool)
        mask[indices] = False
        V_new[mask] = A.mm(V_old)[mask]
        print(V_new[indices].shape, V_new[mask].shape, V_new.shape)
        V_new[indices] = torch.cat(V_new_list, axis=0)[indices]
    return V_new

# 调试
def reconstruction_with_template_2(net, cuda, calib_tensor, resolution, b_min, b_max, save_path, use_octree=False, M=None, landmarks=[], num_samples=10000, transform=None):
    if M == None:
        return
    indices = []
    front = []
    left = []
    right = []
    selected = ["eye-right", "eye-left", "nose", "mouth", "ear-left", "ear-right"]
    with open("/data1/zhongjin/Characters_PIFU/SimpModeling_Colored/checkpoints/pifuhd/hole/new_label_reduced_hole.json", "r") as f:
        label = json.load(f)
        for k, v in label.items():
            if k in selected:
                for i in v:
                    if k == "eye-right" or k == "eye-left" or k == "nose" or k == "mouth":
                        front.append(i)
                    if k == "ear-left":
                        left.append(i)
                    if k == "ear-right":
                        right.append(i)
                    indices.append(i)
    indices = np.concatenate([front, left, right])
    print(len(indices))
    mesh = trimesh.load("/data1/zhongjin/Characters_PIFU/SimpModeling_Colored/checkpoints/pifuhd/hole/template_hole_norm.OBJ", process=False)
    V = torch.from_numpy(mesh.vertices).float()
    F = mesh.faces
    smooth_weight = 1.0
    laplacian_weight = 0.5
    target_weight = 1.0
    A = trimesh.smoothing.laplacian_calculation(mesh)
    I = identity(A.shape[0])
    A = torch.from_numpy(A.toarray()).float()
    I = torch.from_numpy(I.toarray()).float()
    L = (I - A)
    H = torch.zeros((len(indices), L.shape[1]))
    for i in range(H.shape[0]):
        H[i, indices[i]] = 1.0
    D = torch.zeros((L.shape[0], 3))
    D2 = L.mm(V)
    V_bar = torch.from_numpy(landmarks).float()
    L_H = torch.cat([L * smooth_weight, L * laplacian_weight, H * target_weight]).cuda()
    D_V_bar = torch.cat([D * smooth_weight, D2 * laplacian_weight, V_bar * target_weight]).cuda()
    V, _ = torch.lstsq(D_V_bar, L_H)
    V = V[0:D.shape[0]]

    vertices = V.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=F, process=False)
    mesh.export(save_path.replace(".obj", "_01.obj"))

    l = om.TriMesh(points=vertices[indices])
    om.write_mesh(save_path.replace(".obj", "_00.obj"), l)


    smooth_weight = 1.0
    laplacian_weight = 0.00001
    target_weight = 1.0
    A = trimesh.smoothing.laplacian_calculation(mesh)
    I = identity(A.shape[0])
    A = torch.from_numpy(A.toarray()).float().cuda()
    I = torch.from_numpy(I.toarray()).float().cuda()
    L = (I - A)
    H = I.clone().cuda()
    L_H = torch.cat([L * smooth_weight, L * laplacian_weight, H * target_weight])
    V = torch.from_numpy(vertices).float().cuda()
    D = torch.zeros((L.shape[0], 3)).cuda()
    D2 = L.mm(V)
    start = time.time()
    with torch.no_grad():
        for i in range(1):
            D2 = L.mm(V)
            V_bar = get_new_vertices(net, cuda, calib_tensor, V, F, A=None, step = 2.0/(i+1))
            V_bar[indices] = torch.from_numpy(landmarks).float().cuda()
            D_V_bar = torch.cat([D * smooth_weight, D2 * laplacian_weight, V_bar * target_weight])
            V, _ = torch.lstsq(D_V_bar, L_H)
            V = V[0:D.shape[0]]
            vertices = V.detach().cpu().numpy()
            temp = trimesh.Trimesh(vertices=vertices, faces=F, process=False)
            temp.export(save_path.replace(".obj", "_11.obj"))
            l = om.TriMesh(points=vertices[indices])
            om.write_mesh(save_path.replace(".obj", "_10.obj"), l)
            print(i)

    V = A.mm(V)

    vertices = V.detach().cpu().numpy()
    temp = trimesh.Trimesh(vertices=vertices, faces=F, process=False)
    temp.export(save_path.replace(".obj", "_21.obj"))
    l = om.TriMesh(points=vertices[indices])
    om.write_mesh(save_path.replace(".obj", "_20.obj"), l)
    
    # 加细节
    smooth_weight = 1.0
    laplacian_weight = 1.0
    target_weight = 1.0
    L_H = torch.cat([L * smooth_weight, L * laplacian_weight, H * target_weight])
    D = torch.zeros((L.shape[0], 3)).cuda()
    D2 = L.mm(V)
    start = time.time()
    with torch.no_grad():
        for i in range(1):
            vertices = V.detach().cpu().numpy()
            D2 = L.mm(V)
            V_bar = get_new_vertices(net, cuda, calib_tensor, V, F, A=A, step = 0.1/(i+1))
            # V_bar[indices] = torch.from_numpy(landmarks).float().cuda()
            D_V_bar = torch.cat([D * smooth_weight, D2 * laplacian_weight, V_bar * target_weight])
            V, _ = torch.lstsq(D_V_bar, L_H)
            V = V[0:D.shape[0]]
            vertices = V.detach().cpu().numpy()
            temp = trimesh.Trimesh(vertices=vertices, faces=F, process=False)
            temp.export(save_path.replace(".obj", "_31.obj"))
            l = om.TriMesh(points=vertices[indices])
            om.write_mesh(save_path.replace(".obj", "_30.obj"), l)
            print(i)

    vertices = V.detach().cpu().numpy()
    end = time.time()
    print("deformation:", end-start)
    return vertices, F
'''
def reconstruction_with_template_2(net, cuda, calib_tensor, resolution, b_min, b_max, use_octree=False, M=None, landmarks=[], num_samples=10000, transform=None):
    if M == None:
        return
    print("landmarks", landmarks.shape)
    M = np.load('/data1/zhongjin/Characters_PIFU/SimpModeling_Colored/checkpoints/pifuhd/matrix_hole_landmark.npz')
    indices = M["indices"]
    W = M["W"]
    smooth_weight = W[0].item()
    laplacian_weight = W[1].item()
    target_weight = W[2].item()
    R_INV = torch.from_numpy(M["R_INV"]).float().cuda()
    Q_T = torch.from_numpy(M["Q_T"]).float().cuda()
    V = torch.from_numpy(M["V"]).float().cuda()
    F = M["F"]
    L = torch.from_numpy(M["L"]).float().cuda()
    D = torch.zeros((L.shape[0], 3)).cuda()
    D2 = L.mm(V)
    V_bar = torch.from_numpy(landmarks).float().cuda()
    D_V_bar = torch.cat([D * smooth_weight, D2 * laplacian_weight, V_bar * target_weight])
    V = R_INV @ Q_T @ D_V_bar
    V = V[0:D.shape[0]]

    vertices = V.detach().cpu().numpy()
    # mesh = trimesh.Trimesh(vertices=vertices, faces=F, process=False)
    # mesh.export("out.obj")

    M = np.load('/data1/zhongjin/Characters_PIFU/SimpModeling_Colored/checkpoints/pifuhd/matrix_hole_whole.npz')
    W = M["W"]
    smooth_weight = W[0].item()
    laplacian_weight = W[1].item()
    target_weight = W[2].item()
    R_INV = torch.from_numpy(M["R_INV"]).float().cuda()
    Q_T = torch.from_numpy(M["Q_T"]).float().cuda()

    V = torch.from_numpy(vertices).float().cuda()
    F = M["F"]
    L = torch.from_numpy(M["L"]).float().cuda()
    A = torch.from_numpy(M["A"]).float().cuda()
    D = torch.zeros((L.shape[0], 3)).cuda()
    D2 = L.mm(V)
    start = time.time()
    with torch.no_grad():
        for i in range(6):
            vertices = V.detach().cpu().numpy()
            temp = trimesh.Trimesh(vertices=vertices, faces=F, process=False)
            temp.export("outs/" + str(i) + ".obj")
            D2 = L.mm(V)
            V_bar = get_new_vertices(net, cuda, calib_tensor, V, F, A=A, step = 0.5/(i+1))
            V_bar[indices] = torch.from_numpy(landmarks).float().cuda()
            # if i == 5:
            #     V_bar[indices] = torch.from_numpy(landmarks).float().cuda()
            D_V_bar = torch.cat([D * smooth_weight, D2 * laplacian_weight, V_bar * target_weight])
            V = R_INV @ Q_T @ D_V_bar
            V = V[0:D.shape[0]]
            print(i)
    end = time.time()
    print("deformation:", end-start)
    return vertices, F
'''

def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]

        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    
    #print('=== sdf: ', sdf.shape, ' ===')
    #print(sdf)
    # Finally we do marching cubes
    try:
        # print(sdf)
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.0)
        # print("??????")
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()
    # print("OK---------------------")


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
