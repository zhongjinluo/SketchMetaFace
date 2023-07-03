
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
from networks.v0.sketch2norm import Sketch2Norm

app = Flask(__name__, static_folder='', static_url_path='')
s2n = Sketch2Norm()

@app.route('/')
def root():
    return send_from_directory("./", "index.html")

@app.route('/norm_256', methods=["POST"])
def norm_256():
    data = request.get_data()
    json_data = json.loads(data)
    image = np.array(json_data["sketch"], dtype=np.uint8).reshape(480, 480, 3)
    image = Image.fromarray(image).resize((512, 512))
    image.save("test.png")
    s, n = s2n.predict(np.array((image)))
    # s = np.array(Image.fromarray(s).resize((256, 256)), dtype=np.uint8)
    n = np.array(Image.fromarray(n).resize((256, 256)), dtype=np.uint8)
    # n = 0.1 * s + 0.8 * n
    return {
        "norm": n.reshape(-1).tolist(),
        }


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8005)
