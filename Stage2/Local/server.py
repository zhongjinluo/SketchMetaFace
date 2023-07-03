
import re
import io
import os
import time
import json
import trimesh
import base64
import random
import shutil
import numpy as np
from PIL import Image
from flask import Flask, send_from_directory, make_response, jsonify, request

import DracoPy
from networks.v0.sketch2depth import Sketch2Depth
from networks.v0.sketch2norm import Sketch2Norm
from networks.v0.sketch2model_depth import Sketch2Model
from networks.v0.depth2model_flow import Depth2Model_flow
from networks.v0.depth2model_flow_project import Depth2Model_flow_project
from networks.v0.depth2model import Depth2Model

def load_mesh(name):
    mesh = trimesh.load(M0)
    A = trimesh.smoothing.laplacian_calculation(mesh)
    vertices = mesh.vertices
    faces = mesh.faces


    return vertices, faces

def load_img(image):
    image = cv2.imread(image).astype(np.float)
    # note that we need bgr
    image = image[...,::-1]
    image = image/255.
    

    return image

app = Flask(__name__, static_folder='', static_url_path='')
s2n = Sketch2Norm()
s2d = Sketch2Depth()
s2m2 = Sketch2Model()
d2m = Depth2Model()
d2m_flow = Depth2Model_flow()
d2m_project = Depth2Model_flow_project()


def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices, center, r

@app.route('/local', methods=["POST"])
def generate_stage23():
    start = time.time()
    name = str(time.time())

    data = request.get_data()
    json_data = json.loads(data)

    # for sketch
    image = np.array(json_data["front_sketch"], dtype=np.uint8).reshape(480, 480, 3)
    image = Image.fromarray(image).resize((512, 512))
    s = np.array(image)
    sketch_path = "gallery/S1/" + name + ".png"
    image.save(sketch_path)

    # for coarse depth
    depth_path = "gallery/D0/" + name + ".png"
    front_depth = np.array(json_data["side_sketch"], dtype=np.uint8).reshape(480, 480, 3) # side->depth
    front_depth = Image.fromarray(front_depth).resize((512, 512))
    front_depth.save(depth_path)

    # for refined depth
    d0 = np.array(front_depth)
    depth_path = "gallery/RD0/" + name + ".png"
    _, d = s2d.predict(np.array((image)), d0)
    Image.fromarray(d).save(depth_path)
    
    # for normal map
    start_norm = time.time()
    norm_path = "gallery/N/" + name + ".png"
    _, n = s2n.predict(np.array((image)), d0)
    Image.fromarray(n).save(norm_path)
    end = time.time()
    print("normal map:", end - start_norm)

    vertices_list = json_data["vertices_list"]
    faces_list = json_data["faces_list"]

    for i in range(len(vertices_list)):
        if i == 1:
            mesh = trimesh.Trimesh(vertices=vertices_list[i], faces=faces_list[i], process=False)
            mesh.export("gallery/M0/" + name + ".obj")

    start_sdf = time.time()
    obj_path = "gallery/M1/" + name + ".obj"
    faces_list[2] = 2
    vertices_list_xiaojin, faces_list_xiaojin = s2m2.predict_sub_mc(n, d, vertices_list, faces_list, obj_path)
    end = time.time()
    print("sdf deform:", end - start_sdf)
    
    vertices = np.array(vertices_list_xiaojin[2])
    vertices, faces = d2m_flow.predict_from_coarse_model([n.copy()], vertices, np.array(faces_list_xiaojin[1]),  "gallery/HD2/", name)
    vertices[:, 0] += 0.0
    vertices_list_xiaojin[0] = vertices.tolist()
    faces_list_xiaojin[0] = faces.tolist()
    
    vertices = np.array(vertices_list_xiaojin[2])
    vertices, faces = d2m_project.predict_from_coarse_model([n.copy()], vertices, np.array(faces_list_xiaojin[1]),  "gallery/HD2/", name)
    vertices[:, 0] += 0.0
    vertices[:, 1] += 2.0
    vertices_list_xiaojin.append(vertices.tolist())
    faces_list_xiaojin.append(faces.tolist())

    start_depth = time.time()
    vertices = np.array(vertices_list_xiaojin[2])
    vertices, faces = d2m.predict_from_coarse_model([n.copy()], vertices, np.array(faces_list_xiaojin[1]),  "gallery/HD2/", name)
    end_depth = time.time()
    print("depth:", end_depth-start_depth)

    V_temp = np.array(vertices_list_xiaojin[1])
    V_temp[:, 0] -= 1.8
    vertices_list_xiaojin[1] = V_temp.tolist()

    vertices[:, 0] += 1.8
    vertices_list_xiaojin[2] = vertices.tolist()
    faces_list_xiaojin[2] = faces.tolist()


    end = time.time()
    print("all:", end-start)

    return {"vertices": vertices_list_xiaojin, "faces": faces_list_xiaojin}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=39009)