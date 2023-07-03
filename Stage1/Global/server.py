
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
import cv2
import DracoPy
from networks.v0.sketch2norm import Sketch2Norm
from networks.v0.sketch2model import Sketch2Model
from networks.v0.sketch2model2 import Sketch2Model2
from networks.v0.sketch2model3 import Sketch2Model3

app = Flask(__name__, static_folder='', static_url_path='')
s2n = Sketch2Norm()
s2m = Sketch2Model()
s2m2 = Sketch2Model2()
s2m3 = Sketch2Model3()

def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices, center, r

def normalize2(a, b, c):
    mesh_vertices = np.concatenate((np.array(a), np.array(b), np.array(c)),axis=0)
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    return center, r

@app.route('/')
def root():
    return send_from_directory("./", "index.html")

@app.route('/generate', methods=["POST"])
def generate():
    data = request.get_data()
    image = base64.b64decode(re.sub('^data:image/png;base64,', '', data.decode('utf-8')))
    buf = io.BytesIO(image)
    image = Image.open(buf).convert("RGB")
    image = np.array(image)
    # image[image<255] = 0
    image = Image.fromarray(image)
    
    name = str(time.time())
    sketch_path = "gallery/sketch/" + name + ".png"
    image.save(sketch_path)
    norm_path = "gallery/normal/" + name + ".png"
    print(np.array((image)).shape)
    s, n = s2n.predict(np.array((image)))
    n = Image.fromarray(n)
    n.save(norm_path)
    return make_response(jsonify({"normal": norm_path}))

@app.route('/template')
def template():
    template_dir = "sketch_database/"
    images = []
    for f in os.listdir(template_dir):
        images.append({"name": f, "src": template_dir+f})
    images = random.sample(images, 12)
    return make_response(jsonify({"images": images}))

@app.route('/gallery')
def gallery():
    gallery_dir = "gallery/sketch/"
    file_list = []
    for f in os.listdir(gallery_dir):
        file_list.append(f)
    file_list = sorted(file_list)
    images = []
    i = 0
    for f in file_list:
        images.append({"name": f, "src": gallery_dir+f})
        i += 1
    return make_response(jsonify({"images": images[-20:]}))

def get_mask(image):
    image[image[:, :, 1]==0] = 100
    image[image[:, :, 1]>128] = 0
    image[image[:, :, 1]==100] = 255
    lower = (250, 250, 250)
    upper = (255, 255, 255)
    thresh = cv2.inRange(image, lower, upper)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(big_contour)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [big_contour], 0, (255,255,255), -1)
    # cv2.imwrite('frame50_mask.jpg',mask)
    return mask

@app.route('/stage1', methods=["POST"])
def stage2_part():
    data = request.get_data()
    json_data = json.loads(data)
    image = np.array(json_data["front_sketch"], dtype=np.uint8).reshape(480, 480, 3)
    image[image[:, :, 1]<255] = np.array([0, 0, 0])
    image_512 = np.ones((512, 512, 3), dtype=np.uint8) * 255
    image_512[16:496, 16:496, :] = np.array(image)
    # mask = get_mask(np.array(image_512))
    image = Image.fromarray(image_512)
    
    name = str(time.time())
    sketch_path = "gallery/sketch/" + name + ".png"
    image.save(sketch_path)
    
    norm_path = "gallery/normal/" + name + ".png"
    s, n = s2n.predict(np.array((image)))
    Image.fromarray(n).save(norm_path)

    try:
        obj_path = "gallery/obj/" + name + "_part.obj"
        vertices, faces = s2m2.predict(n, obj_path)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        meshes = mesh.split(only_watertight=False)
        if len(meshes) > 0:
            vertices = meshes[0].vertices
            faces = meshes[0].faces
            for m in meshes:
                if m.vertices.shape[0] > vertices.shape[0]:
                    vertices = m.vertices
                    faces = m.faces
        vertices_left = np.array(vertices)
        faces_left = np.array(faces)
        faces_left[:, [0,1,2]] = faces_left[:, [0,2,1]]
        vertices_right = np.array(vertices)
        vertices_right[:, 0] = -vertices_right[:, 0]
        faces_right = np.array(faces)

        obj_path = "gallery/obj/" + name + ".obj"
        vertices, faces = s2m3.predict(n, obj_path)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        meshes = mesh.split(only_watertight=False)
        if len(meshes) > 0:
            vertices = meshes[0].vertices
            faces = meshes[0].faces
            for m in meshes:
                if m.vertices.shape[0] > vertices.shape[0]:
                    vertices = m.vertices
                    faces = m.faces
        faces[:, [0,1,2]] = faces[:, [0,2,1]]
        c, r = normalize2(vertices_left, vertices, vertices_right)
        vertices_left -= c
        vertices_left /= r
        vertices -= c
        vertices /= r
        vertices_right -= c
        vertices_right /= r
        return {"vertices_list": [vertices_left.tolist(), vertices.tolist(), vertices_right.tolist()], "faces_list": [faces_left.tolist(), faces.tolist(), faces_right.tolist()], "layout_shadow": []}
    except Exception as e:
        obj_path = "gallery/obj/" + name + ".obj"
        vertices, faces = s2m.predict(n, obj_path)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        meshes = mesh.split(only_watertight=False)
        if len(meshes) > 0:
            vertices = meshes[0].vertices
            faces = meshes[0].faces
            for m in meshes:
                if m.vertices.shape[0] > vertices.shape[0]:
                    vertices = m.vertices
                    faces = m.faces
        vertices, _, _ = normalize(vertices)
        return {"vertices_list": [[], vertices.tolist(), []], "faces_list": [[], faces.tolist(), []], "layout_shadow": []}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)
