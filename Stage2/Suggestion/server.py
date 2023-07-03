import re
import io
import os
import time
import json
import trimesh
import base64
import cv2
import random
import shutil
import numpy as np
from PIL import Image
from flask import Flask, send_from_directory, make_response, jsonify, request
from get_vec import ImageEmbed

app = Flask(__name__, static_folder='', static_url_path='')

embder = ImageEmbed()

@app.route('/recommend', methods=["POST"])
def recommend():
    name = str(time.time())
    data = request.get_data()
    json_data = json.loads(data)
    image_size = json_data["image_size"]
    smart_type = json_data["smart_type"].lower()
    smart_part = json_data["smart_part"].replace(" ", "_").upper()
    print(smart_type, smart_part)
    sketch = np.array(json_data["part_sketch"], dtype=np.uint8).reshape(image_size, image_size, 3)
    part = np.ones((128, 128, 3)) * 255
    if image_size > 128:
        part = cv2.resize(sketch, (128, 128))
    else:
        s_y = int((128 - image_size) / 2)
        s_x = int((128 - image_size) / 2)
        part[s_y:s_y+image_size, s_x:s_x+image_size] = sketch
    sketch = Image.fromarray(np.uint8(part))
    sketch.save("gallery/B/"+name+".png")
    sketch = np.array(sketch, dtype=np.uint8)
    resultpaths = embder.get_nearest(sketch, [smart_type, smart_part], k_n=10)
    resultpath = resultpaths[0]
    recommend = np.array(Image.open(resultpath).convert('RGB').resize((128, 128)), dtype=np.uint8)
    Image.fromarray(np.uint8(recommend)).save("gallery/RGB/"+name+".png")
    
    recommend_list = []
    gif_paths = []
    for resultpath in resultpaths:
        recommend_img = np.array(Image.open(resultpath).convert('RGB').resize((128, 128)), dtype=np.uint8)
        recommend_list.append(recommend_img.reshape(-1).tolist())
        # gif_paths.append(resultpath.replace(".png", ".gif").replace("STROKES_RGB", "STROKES_GIF"))
        gif_paths.append(resultpath.replace("data/", "").replace(".png", ".png").replace("STROKES_RGB", "STROKES_RGB_BBOX"))
    # print(gif_paths)
    return {
        "recommend": recommend.reshape(-1).tolist(),
        "recommend_list": recommend_list,
        "resultpath_list": gif_paths
        }
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8006)
