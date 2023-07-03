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
import cv2
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from .MyMLP import MyMLP
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

def get_new_vertices_part(net, cuda, calib_tensor, V_old, faces, A = None, flag=False, step=0.018):
    mesh = trimesh.Trimesh(vertices=V_old.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()
    # smooth normal
    if A != None:
        vertex_normals_all = A.mm(vertex_normals_all)
    V_indices = torch.Tensor(range(0, vertex_normals_all.shape[0], 1)).long()
    V_splits = torch.chunk(V_indices, 20, dim=0)
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

        '''
        net.query(V_sample.T.unsqueeze(0), calib_tensor)
        logits = net.get_preds()[0][0]
        logits = torch.abs(logits).reshape(bias.shape[0], -1, 1).squeeze(-1).T
        indices_2 = torch.argmin(logits, 1, keepdim=True).reshape(-1)
        indices_1 = torch.arange(0, logits.shape[0], 1).cuda()
        V_sample =  V_sample.reshape(bias.shape[0], -1, 3)
        V_new = V_sample[indices_2, indices_1, :]
        V_new_list.append(V_new)
        '''

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
    V_new = A.mm(V_new)
    return V_new

# gt -> T0
def get_new_vertices_sub_mc(net, cuda, calib_tensor, V_old, faces, A = None, flag=False, step=0.018, type=0):
    # mesh = trimesh.Trimesh(vertices=V_old.detach().cpu().numpy(), faces=faces, process=False)
    # vertex_normals_all = mesh.vertex_normals
    # vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    # vertex_normals_all = A.dot(vertex_normals_all)
    # vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()
    # net.query(V_old.T.unsqueeze(0), calib_tensor)
    # SDF = net.get_preds()[0][0].reshape(-1, 1)
    # V_new = V_old - step * SDF * vertex_normals_all
    # V_new = torch.from_numpy(A.dot(V_new.detach().cpu().numpy())).float().cuda()
    # return V_new

    mesh = trimesh.Trimesh(vertices=V_old.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = A.dot(vertex_normals_all)
    vertex_normals_all = A.dot(vertex_normals_all)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()

    V_indices = torch.Tensor(range(0, vertex_normals_all.shape[0], 1)).long()
    V_splits = torch.chunk(V_indices, 6, dim=0)
    
    V_new_list = []
    for split in V_splits:
        V = V_old[split] 
        vertex_normals = vertex_normals_all[split]
        net.query(V.T.unsqueeze(0), calib_tensor)
        SDF = net.get_preds()[0][0].reshape(-1, 1)
        V_new = V - step * SDF * vertex_normals
        V_new_list.append(V_new)
    V_new = torch.cat(V_new_list, axis=0)
    V_new = torch.from_numpy(A.dot(V_new.detach().cpu().numpy())).float().cuda()
    indices = np.where(V_new.detach().cpu().numpy()[:, 1] > 0)[0]
    V_new[indices] = torch.from_numpy(A.dot(V_new.detach().cpu().numpy())).float().cuda()[indices]
    V_new[indices] = torch.from_numpy(A.dot(V_new.detach().cpu().numpy())).float().cuda()[indices]
    return V_new

# gt -> T0
def reconstruction_sub_mc(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   vertices_list, faces_list, use_octree=False, num_samples=10000, transform=None):
    iter_count = faces_list[2]
    new_vertices_list = []
    new_vertices_list.append([])
    for p in range(len(vertices_list)):
        if p != 1:
            # new_vertices_list.append([])
            continue
        start = time.time()
        mesh = trimesh.Trimesh(vertices=vertices_list[p], faces=faces_list[p], process=False)
        V = torch.from_numpy(mesh.vertices).float().cuda()
        A = trimesh.smoothing.laplacian_calculation(mesh)
        end = time.time()
        print("A:", end-start)
        start = time.time()
        # A = torch.from_numpy(A.toarray()).float().cuda()
        # A = A.toarray()
        F = mesh.faces
        with torch.no_grad():
            start = time.time()
            for i in range(iter_count):
                if p == 1:
                    if i == 0:
                        V = get_new_vertices_sub_mc(net, cuda, calib_tensor, V, F, A=A, step=0.8, type=0)
                        indices = np.where(V.detach().cpu().numpy()[:, 1] > 0)[0]
                        V[indices] = torch.from_numpy(A.dot(V.detach().cpu().numpy())).float().cuda()[indices]
                    else:
                        V = torch.from_numpy(mesh.vertices).float().cuda()
                        V = get_new_vertices_sub_mc(net, cuda, calib_tensor, V, F, A=A, step=0.6, type=0)
                        V[indices] = torch.from_numpy(A.dot(V.detach().cpu().numpy())).float().cuda()[indices]
                        V[indices] = torch.from_numpy(A.dot(V.detach().cpu().numpy())).float().cuda()[indices]
                        V = get_new_vertices_sub_mc(net, cuda, calib_tensor, V, F, A=A, step=0.8, type=0)
                        V[indices] = torch.from_numpy(A.dot(V.detach().cpu().numpy())).float().cuda()[indices]
                        V[indices] = torch.from_numpy(A.dot(V.detach().cpu().numpy())).float().cuda()[indices]
                        # V = get_new_vertices_sub_mc(net, cuda, calib_tensor, V, F, A=A, step=0.98, type=0)
                        # V[indices] = torch.from_numpy(A.dot(V.detach().cpu().numpy())).float().cuda()[indices]
                else:
                    pass
                vertices = V.detach().cpu().numpy()
                new_vertices_list.append(vertices.tolist())
                # end = time.time()
                # print("deformation1:", end-start)
                # print("deformation2:", end-start)
            if p == 1:
                end = time.time()
                print("deformation:", end-start)
        # vertices = V.detach().cpu().numpy()
        # new_vertices_list.append(vertices.tolist())
    return new_vertices_list, faces_list, None, None

def depth2mesh(data_uint16, save_path, image_size=512):
    def unproject(points, calibrations, transforms=None):
        rot = calibrations[:, :3, :3]
        trans = calibrations[:, :3, 3:4]
        pts = torch.bmm(rot, points)
        pts = trans + pts
        return pts
    with open("networks/v0/pifu_sdf_DeepImplicitTemplates_Depth/PV.json", 'r') as f:
        data = json.load(f)
        P = np.array(data["P"]).reshape(-1, 4)
        P[1, 1] = -P[1, 1]
        V = np.array(data["V"]).reshape(-1, 4)
        calib = torch.from_numpy(P.dot(V)).float()
    data_uint16 = cv2.resize(data_uint16, (image_size,image_size))
    dep_map = data_uint16.copy()
    img_h, img_w = dep_map.shape
    h, w = dep_map.shape
    vid, uid = np.where(dep_map < 250*250)
    nv = len(vid)

    ### calculate the inverse point cloud
    uv_mat = np.ones((nv, 3), dtype=np.float16)
    uv_mat[:, 0] = (uid - img_h/2.)/img_h*2.
    uv_mat[:, 1] = (img_h/2. - vid)/img_h*2.
    points = torch.from_numpy(uv_mat.T).float()
    P[1, 1] = -P[1, 1]
    calib = torch.from_numpy(P.dot(V)).float()
    calib = torch.inverse(calib)
    pts = unproject(points.unsqueeze(0), calib.unsqueeze(0))
    vert = pts[0, :3, :].T
    vert = vert.detach().cpu().numpy()
    vert[:, 2] = dep_map[vid, uid]/255./255.
    vert[:, 2] = vert[:, 2] * 2 - 1
    vert[:, 2] = dep_map[vid, uid]/255./255.
    vert[:, 2] = vert[:, 2] * 2 - 1
    
    f = open(save_path, 'w')
    print(save_path)
    nv = 0
    vidx_map = np.full_like(dep_map, fill_value=-1, dtype=np.int)
    for i in range(0, len(vid)):
        f.write('v %f %f %f\n' % (vert[i][0], vert[i][1], vert[i][2]))
        vidx_map[vid[i], uid[i]] = nv
        nv += 1
    for i in range(0, h-2):
        for j in range(0, w-2):
            if vidx_map[i, j] >= 0 and vidx_map[i, j+1] >= 0 and vidx_map[i+1, j] >= 0 and vidx_map[i+1, j+1] >= 0:
                f.write('f %d %d %d\n' % (vidx_map[i , j] + 1, vidx_map[i+1, j] + 1, vidx_map[i, j+1] + 1))
                f.write('f %d %d %d\n' % (vidx_map[i + 1, j + 1] + 1, vidx_map[i, j+1] + 1, vidx_map[i+1, j] + 1))
    f.close()
    
    return vid, uid, vert


def get_z_from_depth(vertices, vertex_normals, A, depth):
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    def get_indices(mesh_normal, mesh_vertices):
        mesh_normal = unit_vector(mesh_normal)
        z_axis = np.array([0, 0, 1])
        flags = np.dot(mesh_normal, z_axis)
        indices0 = np.where(flags > 0.001)[0]
        indices = []
        for i in indices0:
            if mesh_vertices[i, 2] > 0:
                indices.append(i)
        return indices

    def project(points, calibrations):
        rot = calibrations[:, :3, :3]
        trans = calibrations[:, :3, 3:4]
        pts = torch.baddbmm(trans, rot, points)
        return pts

    with open("networks/v0/pifu_sdf_DeepImplicitTemplates_Depth/PV.json", 'r') as f:
        data = json.load(f)
        P = np.array(data["P"]).reshape(-1, 4)
        P[1, 1] = -P[1, 1]
        V = np.array(data["V"]).reshape(-1, 4)
        calib = torch.from_numpy(P.dot(V)).float()
    
    front_indices = get_indices(vertex_normals, vertices)
    points = torch.from_numpy(vertices[front_indices].T).float()
    pts = project(points.unsqueeze(0), calib.unsqueeze(0))
    xy = pts[:, :2, :]
    XY = (xy[0] + 1) * 512 * 0.5
    XY = XY.numpy().astype(int)

    positions = depth[XY[1], XY[0]] / 255.0 / 255.0
    temp = vertices[front_indices]
    temp[:, 2] = positions
    temp[:, 2] = temp[:, 2] * 2 - 1
    vertices[front_indices] = temp

    return A.dot(vertices), front_indices, np.array(vertices[:, 2])


def get_feature(vertices):
    def project_feature(points, calibrations):
        rot = calibrations[:, :3, :3]
        trans = calibrations[:, :3, 3:4]
        pts = torch.baddbmm(trans, rot, points)
        return pts

    def index(feat, uv):
        '''

        :param feat: [B, C, H, W] image features
        :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
        :return: [B, C, N] image features at the uv coordinates
        '''
        uv = uv.transpose(1, 2)  # [B, N, 2]
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
        # for old versions, simply remove the aligned_corners argument.
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    with open("networks/v0/pifu_sdf_DeepImplicitTemplates_Depth/PV.json", 'r') as f:
        data = json.load(f)
        P = np.array(data["P"]).reshape(-1, 4)
        P[1, 1] = -P[1, 1]
        V = np.array(data["V"]).reshape(-1, 4)
        calib = torch.from_numpy(P.dot(V)).float()

    to_tensor = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    to_tensor_d = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    points = torch.from_numpy(vertices.T).float()
    fnorm = Image.open("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER3/DE_TEST/N/RAW_mesh_Amanda+Seyfried_C00013.png").convert('RGB')
    fdepth = cv2.imread("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER3/DE_TEST/RD/RAW_mesh_Amanda+Seyfried_C00013.png", -1)
    fdepth = np.array(fdepth/255.0/255.0).reshape(512, 512, 1)
    fnorm = to_tensor(fnorm)
    fdepth = to_tensor_d(fdepth).float()
    render = torch.cat([fnorm, fdepth], 0)
    im_feat = render.unsqueeze(0).cuda()
    pts = project_feature(points.unsqueeze(0).cuda(), calib.unsqueeze(0).cuda())
    xy = pts[:, :2, :]
    feature = index(im_feat, xy)

    # print(feature.shape)
    check = np.ones((512, 512, 1))
    XY = (xy[0] + 1) * 256
    XY = XY.detach().cpu().numpy().astype(int)
    pixels = index(im_feat, xy)[0][0].detach().cpu().numpy()
    for i in range(len(pixels)):
        check[XY[1, i], XY[0, i]] = pixels[i]
    cv2.imwrite("check.png", check*255)

    # print(im_feat[0].permute(2, 1, 0).shape)
    img = im_feat[0].permute(1, 2, 0)[:, :, :3].detach().cpu().numpy()
    img = (img + 1) / 2 * 255.0
    # out = img.reshape(512, 512, 1)
    cv2.imwrite("im_feat.png", img)
    # xx
    # print(feature.shape, points.unsqueeze(0).shape)
    feature = torch.cat([feature, points.unsqueeze(0).cuda()], 1)
    # xx
    return feature
'''
def get_new_vertices_de(net, cuda, calib_tensor, V_old, faces, A = None, flag=False, step=0.018, type=0):
    mesh = trimesh.Trimesh(vertices=V_old.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = A.dot(vertex_normals_all)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()
    net.query(V_old.T.unsqueeze(0), calib_tensor)
    SDF = net.get_preds()[0][0].reshape(-1, 1)
    V_new = V_old - step * SDF * vertex_normals_all
    V_new = torch.from_numpy(A.dot(V_new.detach().cpu().numpy())).float().cuda()
    return V_new

def get_new_vertices_sdf_de(net, cuda, calib_tensor, V_old, faces, A = None, rd=None, is_de=False, step=1.0, type=0):
    mesh = trimesh.Trimesh(vertices=np.array(V_old.detach().cpu().numpy()), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = A.dot(vertex_normals_all)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()


    feature = get_feature(mesh.vertices)
    

    vertices = np.array(V_old.detach().cpu().numpy())
    temp_v, indices, _ = get_z_from_depth(vertices, mesh.vertex_normals, A, rd)
    DELTA_D_Z =  temp_v - mesh.vertices
    DELTA_D_Z = torch.from_numpy(DELTA_D_Z).float().cuda()

    # for 
    net.query(V_old.T.unsqueeze(0), calib_tensor)
    SDF = net.get_preds()[0][0].reshape(-1, 1)
    SDF_N = -SDF * vertex_normals_all
    # SDF_N = torch.from_numpy(A.dot(SDF_N.detach().cpu().numpy())).float().cuda()
    
    gt = trimesh.load("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER3/DE_TEST/Tn/RAW_mesh_Amanda+Seyfried_C00013.obj", process=False)
    V_gt = torch.from_numpy(gt.vertices).float().cuda()
    
    SDF_N = torch.from_numpy(SDF_N.detach().cpu().numpy()).float().cuda()

    # W = torch.full((V_gt.shape[0], 3), 1.0)
    # W = W.cuda()
    # W.requires_grad = True
    # print(SDF_N.requires_grad)
    mlp = MyMLP(7, 256, 3).cuda().train()
    # optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0003)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.0003, momentum=0.9)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(10000):
        W = mlp(feature.squeeze(0).T)
        DELTA = (1 - W) * DELTA_D_Z + W * SDF_N
        V_new = V_old + DELTA
        # print(DELTA)
        error = loss_fn(V_new, V_gt) * 10.0

        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        print(epoch, error.item())
    temp = trimesh.Trimesh(vertices=V_gt.detach().cpu().numpy(), faces=mesh.faces, process=False)
    colors, _, _ = normalize(W.detach().cpu().numpy())
    colors = (colors + 1) * 0.5
    colors *= 255
    temp.visual.vertex_colors = colors
    temp.export("xyz.obj")

    colors_single = np.zeros(colors.shape)
    colors_single[:, 0] = colors[:, 0]
    temp.visual.vertex_colors = colors_single
    temp.export("x.obj")

    colors_single = np.zeros(colors.shape)
    colors_single[:, 1] = colors[:, 1]
    temp.visual.vertex_colors = colors_single
    temp.export("y.obj")

    colors_single = np.zeros(colors.shape)
    colors_single[:, 2] = colors[:, 2]
    temp.visual.vertex_colors = colors_single
    temp.export("z.obj")

    colors_single = np.zeros(colors.shape)
    i = 0
    for w in W[:, 0]:
        if abs(w) > abs(1 - w):
            colors_single[i, 0] = 255
        else:
            colors_single[i, [0,1,2]] = 255
        i += 1
    temp.visual.vertex_colors = colors_single
    temp.export("x_s.obj")

    colors_single = np.zeros(colors.shape)
    i = 0
    for w in W[:, 1]:
        if abs(w) > abs(1 - w):
            colors_single[i, 1] = 255
        else:
            colors_single[i, [0,1,2]] = 255
        i += 1
    temp.visual.vertex_colors = colors_single
    temp.export("y_s.obj")

    colors_single = np.zeros(colors.shape)
    i = 0
    for w in W[:, 2]:
        if abs(w) > abs(1 - w):
            colors_single[i, 2] = 255
        else:
            colors_single[i, [0,1,2]] = 255
        i += 1
    temp.visual.vertex_colors = colors_single
    temp.export("z_s.obj")

    print(W)
    return V_new
'''
# opt weight
'''
def get_new_vertices_sdf_de(net, cuda, calib_tensor, V_old, faces, A = None, rd=None, is_de=False, step=1.0, type=0):
    mesh = trimesh.Trimesh(vertices=np.array(V_old.detach().cpu().numpy()), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = A.dot(vertex_normals_all)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()

    vertices = np.array(V_old.detach().cpu().numpy())
    temp_v, indices, _ = get_z_from_depth(vertices, mesh.vertex_normals, A, rd)
    DELTA_D_Z =  temp_v - mesh.vertices
    DELTA_D_Z = torch.from_numpy(DELTA_D_Z).float().cuda()

    # for 
    net.query(V_old.T.unsqueeze(0), calib_tensor)
    SDF = net.get_preds()[0][0].reshape(-1, 1)
    SDF_N = -SDF * vertex_normals_all
    # SDF_N = torch.from_numpy(A.dot(SDF_N.detach().cpu().numpy())).float().cuda()
    
    gt = trimesh.load("/program/SIGRAPH22/APP_V3/SimpModeling_Depth_DIT_RENDER3/DE_TEST/Tn/RAW_mesh_Amanda+Seyfried_C00013.obj", process=False)
    V_gt = torch.from_numpy(gt.vertices).float().cuda()
    
    SDF_N = torch.from_numpy(SDF_N.detach().cpu().numpy()).float().cuda()

    W = torch.full((V_gt.shape[0], 3), 1.0)
    W = W.cuda()
    W.requires_grad = True
    # print(SDF_N.requires_grad)
    optimizer = torch.optim.Adam([W], lr=1.0)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(1000):
        DELTA = (1 - W) * DELTA_D_Z + W * SDF_N
        V_new = V_old + DELTA
        # print(DELTA)
        error = loss_fn(V_new, V_gt) * 1000.0

        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        print(epoch, error.item())
    temp = trimesh.Trimesh(vertices=V_gt.detach().cpu().numpy(), faces=mesh.faces, process=False)
    colors, _, _ = normalize(W.detach().cpu().numpy())
    colors = (colors + 1) * 0.5
    colors *= 255
    temp.visual.vertex_colors = colors
    temp.export("xyz.obj")

    colors_single = np.zeros(colors.shape)
    colors_single[:, 0] = colors[:, 0]
    temp.visual.vertex_colors = colors_single
    temp.export("x.obj")

    colors_single = np.zeros(colors.shape)
    colors_single[:, 1] = colors[:, 1]
    temp.visual.vertex_colors = colors_single
    temp.export("y.obj")

    colors_single = np.zeros(colors.shape)
    colors_single[:, 2] = colors[:, 2]
    temp.visual.vertex_colors = colors_single
    temp.export("z.obj")

    colors_single = np.zeros(colors.shape)
    i = 0
    for w in W[:, 0]:
        if abs(w) > abs(1 - w):
            colors_single[i, 0] = 255
        else:
            colors_single[i, [0,1,2]] = 255
        i += 1
    temp.visual.vertex_colors = colors_single
    temp.export("x_s.obj")

    colors_single = np.zeros(colors.shape)
    i = 0
    for w in W[:, 1]:
        if abs(w) > abs(1 - w):
            colors_single[i, 1] = 255
        else:
            colors_single[i, [0,1,2]] = 255
        i += 1
    temp.visual.vertex_colors = colors_single
    temp.export("y_s.obj")

    colors_single = np.zeros(colors.shape)
    i = 0
    for w in W[:, 2]:
        if abs(w) > abs(1 - w):
            colors_single[i, 2] = 255
        else:
            colors_single[i, [0,1,2]] = 255
        i += 1
    temp.visual.vertex_colors = colors_single
    temp.export("z_s.obj")

    print(W)
    return V_new
'''

def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices, center, r

def reconstruction_de(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   vertices_list, faces_list, use_octree=False, num_samples=10000, transform=None):
    save_path = faces_list[0]
    rd = faces_list[2]
    

    depth2mesh(rd, save_path[0:-4] + "_d2m.obj", image_size=512)

    new_vertices_list = []
    for p in range(len(vertices_list)):
        if p != 1:
            new_vertices_list.append([])
            continue
        start = time.time()
        mesh = trimesh.Trimesh(vertices=vertices_list[p], faces=faces_list[p], process=False)
        A = trimesh.smoothing.laplacian_calculation(mesh)
        A = A.toarray()
        mesh.export(save_path[0:-4] + "_T0.obj")

        # only delta_d * z
        vertices = np.array(mesh.vertices)
        temp_v, _, _ = get_z_from_depth(vertices, mesh.vertex_normals, A, rd)
        temp = trimesh.Trimesh(vertices=temp_v, faces=faces_list[p], process=False)
        temp.export(save_path[0:-4] + "_de.obj")

        vertices = np.array(mesh.vertices)
        V = torch.from_numpy(vertices).float().cuda()
        F = mesh.faces
        temp = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=mesh.faces, process=False)
        temp.export(save_path[0:-4] + "_" + str(0) + ".obj")
        for i in range(1):
            if p == 1:
                if i < 2:
                    V = get_new_vertices_sdf_de(net, cuda, calib_tensor, V, F, A=A, rd=rd, is_de=True, step=1.0/(i+1.0), type=0)
                else:
                    V = get_new_vertices_sdf_de(net, cuda, calib_tensor, V, F, A=A, rd=rd, is_de=True, step=1.0/(i+1.0), type=0)
                temp = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=mesh.faces, process=False)
                temp.export(save_path[0:-4] + "_" + str(i+1) + ".obj")
        temp = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=mesh.faces, process=False)
        temp.export(save_path[0:-4] + "_sdf_de.obj")

        # only sdf * n
        vertices = np.array(mesh.vertices)
        V = torch.from_numpy(vertices).float().cuda()
        F = mesh.faces
        with torch.no_grad():
            start = time.time()
            for i in range(2):
                if p == 1:
                    if i == 0:
                        V = get_new_vertices_de(net, cuda, calib_tensor, V, F, A=A, step=1.0, type=0)
                    else:
                        V = get_new_vertices_de(net, cuda, calib_tensor, V, F, A=A, step=1.0, type=0)
                else:
                    pass
            if p == 1:
                end = time.time()
                print("deformation:", end-start)
        vertices = V.detach().cpu().numpy()
        temp = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
        temp.export(save_path[0:-4] + "_sdf_df.obj")

        vertices = np.array(mesh.vertices)
        temp_v, _, _ = get_z_from_depth(vertices, temp.vertex_normals, A, rd)
        temp = trimesh.Trimesh(vertices=temp_v, faces=faces_list[p], process=False)
        temp.export(save_path[0:-4] + "_de2.obj")

        new_vertices_list.append(vertices.tolist())

    return new_vertices_list, faces_list, None, None

''' general
def get_new_vertices_sub_mc(net, cuda, calib_tensor, V_old, faces, A = None, flag=False, step=0.018, type=0):
    mesh = trimesh.Trimesh(vertices=V_old.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()
    # smooth normal
    if A != None and type == 0:
        vertex_normals_all = A.mm(vertex_normals_all)
        net.query(V_old.T.unsqueeze(0), calib_tensor)
        SDF = net.get_preds()[0][0].reshape(-1, 1)
        V_new = V_old - step * SDF * vertex_normals_all
        V_new = A.mm(V_new)
    elif type == 1:
        net.query(V_old.T.unsqueeze(0), calib_tensor)
        SDF = net.get_preds()[0][0].reshape(-1, 1)
        # V_new = V_old - step * SDF * V_old
        V_new = V_old
        signs = torch.sign(vertex_normals_all[:, 2])
        indices = torch.where(signs>0)
        V_new[indices] = (V_old - step * SDF * V_old)[indices]
    else:
        directions = torch.Tensor([[0.0, 0.0, 1.0]]).cuda()
        signs = torch.sign(vertex_normals_all[:, 2])
        directions = signs.reshape(-1, 1) * directions.repeat(vertex_normals_all.shape[0], 1)
        net.query(V_old.T.unsqueeze(0), calib_tensor)
        SDF = net.get_preds()[0][0].reshape(-1, 1)
        indices = torch.where(signs>0)
        V_new = V_old
        V_new[indices] = (V_old - step * SDF * directions)[indices]
    return V_new

def reconstruction_sub_mc(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   vertices_list, faces_list, use_octree=False, num_samples=10000, transform=None):
    new_vertices_list = []
    for p in range(len(vertices_list)):
        if p != 1:
            new_vertices_list.append([])
            continue
        start = time.time()
        mesh = trimesh.Trimesh(vertices=vertices_list[p], faces=faces_list[p], process=False)
        V = torch.from_numpy(mesh.vertices).float().cuda()
        A = trimesh.smoothing.laplacian_calculation(mesh)
        A = torch.from_numpy(A.toarray()).float().cuda()
        # A = A.toarray()
        F = mesh.faces
        with torch.no_grad():
            start = time.time()
            for i in range(2):
                if p == 1:
                    if i == 0:
                        V = get_new_vertices_sub_mc(net, cuda, calib_tensor, V, F, A=A, step=1.0, type=0)
                    else:
                        V = get_new_vertices_sub_mc(net, cuda, calib_tensor, V, F, A=A, step=1.0, type=0)
                else:
                    pass
            if p == 1:
                end = time.time()
                print("deformation:", end-start)
        vertices = V.detach().cpu().numpy()
        new_vertices_list.append(vertices.tolist())
    return new_vertices_list, faces_list, None, None
'''



def depth2mesh(data_uint16, save_path, image_size=512):
    def unproject(points, calibrations, transforms=None):
        rot = calibrations[:, :3, :3]
        trans = calibrations[:, :3, 3:4]
        pts = torch.bmm(rot, points)
        pts = trans + pts
        return pts
    with open("networks/v0/pifu_sdf_DeepImplicitTemplates_Depth/PV.json", 'r') as f:
        data = json.load(f)
        P = np.array(data["P"]).reshape(-1, 4)
        P[1, 1] = -P[1, 1]
        V = np.array(data["V"]).reshape(-1, 4)
        calib = torch.from_numpy(P.dot(V)).float()
    data_uint16 = cv2.resize(data_uint16, (image_size,image_size))
    dep_map = data_uint16.copy()
    img_h, img_w = dep_map.shape
    h, w = dep_map.shape
    vid, uid = np.where(dep_map < 250*250)
    nv = len(vid)

    ### calculate the inverse point cloud
    uv_mat = np.ones((nv, 3), dtype=np.float16)
    uv_mat[:, 0] = (uid - img_h/2.)/img_h*2.
    uv_mat[:, 1] = (img_h/2. - vid)/img_h*2.
    points = torch.from_numpy(uv_mat.T).float()
    P[1, 1] = -P[1, 1]
    calib = torch.from_numpy(P.dot(V)).float()
    calib = torch.inverse(calib)
    pts = unproject(points.unsqueeze(0), calib.unsqueeze(0))
    vert = pts[0, :3, :].T
    vert = vert.detach().cpu().numpy()
    vert[:, 2] = dep_map[vid, uid]/255./255.
    vert[:, 2] = vert[:, 2] * 2 - 1
    vert[:, 2] = dep_map[vid, uid]/255./255.
    vert[:, 2] = vert[:, 2] * 2 - 1
    
    f = open(save_path, 'w')
    print(save_path)
    nv = 0
    vidx_map = np.full_like(dep_map, fill_value=-1, dtype=np.int)
    for i in range(0, len(vid)):
        f.write('v %f %f %f\n' % (vert[i][0], vert[i][1], vert[i][2]))
        vidx_map[vid[i], uid[i]] = nv
        nv += 1
    for i in range(0, h-2):
        for j in range(0, w-2):
            if vidx_map[i, j] >= 0 and vidx_map[i, j+1] >= 0 and vidx_map[i+1, j] >= 0 and vidx_map[i+1, j+1] >= 0:
                f.write('f %d %d %d\n' % (vidx_map[i , j] + 1, vidx_map[i+1, j] + 1, vidx_map[i, j+1] + 1))
                f.write('f %d %d %d\n' % (vidx_map[i + 1, j + 1] + 1, vidx_map[i, j+1] + 1, vidx_map[i+1, j] + 1))
    f.close()
    
    return vid, uid, vert


def get_z_from_depth(vertices, vertex_normals, A, depth):
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    def get_indices(mesh_normal, mesh_vertices):
        mesh_normal = unit_vector(mesh_normal)
        z_axis = np.array([0, 0, 1])
        flags = np.dot(mesh_normal, z_axis)
        indices0 = np.where(flags > 0.001)[0]
        indices = []
        for i in indices0:
            if mesh_vertices[i, 2] > 0:
                indices.append(i)
        return indices

    def project(points, calibrations):
        rot = calibrations[:, :3, :3]
        trans = calibrations[:, :3, 3:4]
        pts = torch.baddbmm(trans, rot, points)
        return pts

    with open("networks/v0/pifu_sdf_DeepImplicitTemplates_Depth/PV.json", 'r') as f:
        data = json.load(f)
        P = np.array(data["P"]).reshape(-1, 4)
        P[1, 1] = -P[1, 1]
        V = np.array(data["V"]).reshape(-1, 4)
        calib = torch.from_numpy(P.dot(V)).float()
    
    front_indices = get_indices(vertex_normals, vertices)
    points = torch.from_numpy(vertices[front_indices].T).float()
    pts = project(points.unsqueeze(0), calib.unsqueeze(0))
    xy = pts[:, :2, :]
    XY = (xy[0] + 1) * 512 * 0.5
    XY = XY.numpy().astype(int)

    positions = depth[XY[1], XY[0]] / 255.0 / 255.0
    temp = vertices[front_indices]
    temp[:, 2] = positions
    temp[:, 2] = temp[:, 2] * 2 - 1
    vertices[front_indices] = temp

    return A.dot(vertices), front_indices, np.array(vertices[:, 2])

def get_new_vertices_de(net, cuda, calib_tensor, V_old, faces, A = None, flag=False, step=0.018, type=0):
    mesh = trimesh.Trimesh(vertices=V_old.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = A.dot(vertex_normals_all)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()
    net.query(V_old.T.unsqueeze(0), calib_tensor)
    SDF = net.get_preds()[0][0].reshape(-1, 1)
    V_new = V_old - step * SDF * vertex_normals_all
    V_new = torch.from_numpy(A.dot(V_new.detach().cpu().numpy())).float().cuda()
    return V_new

def get_new_vertices_sdf_de(net, cuda, calib_tensor, V_old, faces, A = None, rd=None, is_de=False, step=1.0, type=0):
    mesh = trimesh.Trimesh(vertices=np.array(V_old.detach().cpu().numpy()), faces=faces, process=False)
    vertex_normals_all = mesh.vertex_normals
    vertex_normals_all = vertex_normals_all / np.sqrt(np.sum(np.array(vertex_normals_all**2), axis=-1)).reshape(-1, 1)
    vertex_normals_all = A.dot(vertex_normals_all)
    vertex_normals_all = torch.from_numpy(vertex_normals_all).float().cuda()

    if not is_de:
        net.query(V_old.T.unsqueeze(0), calib_tensor)
        SDF = net.get_preds()[0][0].reshape(-1, 1)
        V_new = V_old - step * SDF * vertex_normals_all
        # V_new = torch.from_numpy(A.dot(V_new.detach().cpu().numpy())).float().cuda()
    else:
        vertices = np.array(V_old.detach().cpu().numpy())
        temp_v, indices, _ = get_z_from_depth(vertices, mesh.vertex_normals, A, rd)
        DELTA_D_Z =  temp_v - mesh.vertices
        DELTA_D_Z = torch.from_numpy(DELTA_D_Z).float().cuda()

        net.query(V_old.T.unsqueeze(0), calib_tensor)
        SDF = net.get_preds()[0][0].reshape(-1, 1)
        SDF_N = -SDF * vertex_normals_all
        # SDF_N = torch.from_numpy(A.dot(SDF_N.detach().cpu().numpy())).float().cuda()

        # W = torch.ones((mesh.vertices.shape[0], 3)).float().cuda()
        # W[indices, :] = 1.0
        # print(step)
        DELTA = (1 - step) * DELTA_D_Z + step * SDF_N
        # torch.from_numpy(A.dot(V_new.detach().cpu().numpy())).float().cuda()
        V_new = V_old + DELTA
        # V_new = torch.from_numpy(A.dot(V_new.detach().cpu().numpy())).float().cuda()
    return V_new

def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices, center, r

def reconstruction_de(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   vertices_list, faces_list, use_octree=False, num_samples=10000, transform=None):
    save_path = faces_list[0]
    rd = faces_list[2]
    

    # d2m
    depth2mesh(rd, save_path[0:-4] + "_d2m.obj", image_size=512)

    new_vertices_list = []
    for p in range(len(vertices_list)):
        if p != 1:
            new_vertices_list.append([])
            continue
        start = time.time()
        mesh = trimesh.Trimesh(vertices=vertices_list[p], faces=faces_list[p], process=False)
        A = trimesh.smoothing.laplacian_calculation(mesh)
        A = A.toarray()
        mesh.export(save_path[0:-4] + "_T0.obj")

        # only delta_d * z
        vertices = np.array(mesh.vertices)
        temp_v, _, _ = get_z_from_depth(vertices, mesh.vertex_normals, A, rd)
        temp = trimesh.Trimesh(vertices=temp_v, faces=faces_list[p], process=False)
        temp.export(save_path[0:-4] + "_de.obj")

        vertices = np.array(mesh.vertices)
        V = torch.from_numpy(vertices).float().cuda()
        F = mesh.faces
        with torch.no_grad():
            temp = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=mesh.faces, process=False)
            temp.export(save_path[0:-4] + "_" + str(0) + ".obj")
            for i in range(6):
                if p == 1:
                    if i < 2:
                        V = get_new_vertices_sdf_de(net, cuda, calib_tensor, V, F, A=A, rd=rd, is_de=True, step=1.0/(i+1.0), type=0)
                    else:
                        V = get_new_vertices_sdf_de(net, cuda, calib_tensor, V, F, A=A, rd=rd, is_de=True, step=1.0/(i+1.0), type=0)
                    temp = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=mesh.faces, process=False)
                    temp.export(save_path[0:-4] + "_" + str(i+1) + ".obj")
            temp = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=mesh.faces, process=False)
            temp.export(save_path[0:-4] + "_sdf_de.obj")

        # only sdf * n
        vertices = np.array(mesh.vertices)
        V = torch.from_numpy(vertices).float().cuda()
        F = mesh.faces
        with torch.no_grad():
            start = time.time()
            for i in range(2):
                if p == 1:
                    if i == 0:
                        V = get_new_vertices_de(net, cuda, calib_tensor, V, F, A=A, step=1.0, type=0)
                    else:
                        V = get_new_vertices_de(net, cuda, calib_tensor, V, F, A=A, step=1.0, type=0)
                else:
                    pass
            if p == 1:
                end = time.time()
                print("deformation:", end-start)
        vertices = V.detach().cpu().numpy()
        temp = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
        temp.export(save_path[0:-4] + "_sdf_df.obj")

        vertices = np.array(mesh.vertices)
        temp_v, _, _ = get_z_from_depth(vertices, temp.vertex_normals, A, rd)
        temp = trimesh.Trimesh(vertices=temp_v, faces=faces_list[p], process=False)
        temp.export(save_path[0:-4] + "_de2.obj")

        new_vertices_list.append(vertices.tolist())

    return new_vertices_list, faces_list, None, None


def reconstruction_part(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   vertices_list, faces_list, use_octree=False, num_samples=10000, transform=None):
    new_vertices_list = []
    start = time.time()
    for p in range(len(vertices_list)):
        if p != 1:
            new_vertices_list.append([])
            continue
        mesh = trimesh.Trimesh(vertices=vertices_list[p], faces=faces_list[p], process=False)
        V = torch.from_numpy(mesh.vertices).float().cuda()
        F = mesh.faces
        A = torch.from_numpy(trimesh.smoothing.laplacian_calculation(mesh).toarray()).float().cuda()
        with torch.no_grad():
            for i in range(3):
                if p == 1:
                    V = get_new_vertices_part(net, cuda, calib_tensor, V, F, A=A, step = 0.6/(i+1))
                else:
                    V = get_new_vertices_part(net, cuda, calib_tensor, V, F, A=A, step = 0.3/(i+1))
                    # pass
        end = time.time()
        vertices = V.detach().cpu().numpy()
        new_vertices_list.append(vertices.tolist())
    print("deformation:", end-start)
    return new_vertices_list, faces_list, None, None

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
        for i in range(6):
            D2 = L.mm(V)
            V_bar = get_new_vertices(net, cuda, calib_tensor, V, F, A=None, step = 0.5/(i+1))
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
    V_splits = torch.chunk(V_indices, 1, dim=0)
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
        print("eval_grid_octree(")
        start = time.time()
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
        end = time.time()
        print(end - start)
    else:
        print("eval_grid")
        start = time.time()
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
        end = time.time()
        print(end - start)
    
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
