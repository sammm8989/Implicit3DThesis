from flask import Flask, request, jsonify, Response
from PIL import Image
from process_API import run, initiate
import argparse
from configs.config_utils import CONFIG
import io
import numpy as np
from time import time
import json
import gzip




parser = argparse.ArgumentParser('Total 3D Understanding.')
parser.add_argument('config', type=str, default='configs/total3d_mgnet.yaml',
                    help='configure file for training or testing.')
parser.add_argument('--mode', type=str, default='demo', help='train, test, demo_with_time, demo or qtrain, qtest')
parser.add_argument('--demo_path', type=str, default='demo/inputs/1', help='Please specify the demo path.')
parser.add_argument('--save_results', type=str, default='demo/data_time/')
parser.add_argument('--name', type=str, default=None, help='wandb exp name.')
parser.add_argument('--avg_amount', type=int, default=None, help='The amount of samples to run the timing on')
parser.add_argument('--sweep', action='store_true')
cfg = CONFIG(parser)
cfg.config['mode'] = 'demo'
initiate(cfg)

preprocess = []
calculations = []

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_photo():
    start = time()
    if 'photo' not in request.files:
        return jsonify({'error': 'No file part'})
    if 'cam' not in request.files:
        return jsonify({'error': 'No file part'})
    
    photo = request.files['photo']
    cam = request.files["cam"]

    buffer_content = cam.read()

    content_str = buffer_content.decode('utf-8')
    data = [[float(num) for num in line.split()] for line in content_str.split('\n') if line.strip()]
    array = np.array(data)

    stream = io.BytesIO(photo.read())
    image = Image.open(stream).convert("RGB")

    preprocess.append(time() - start)
    #print data to asses response time
    #print("preprocess = " + str(preprocess))
   
    compressed_data = run(image,array)
     
    return compressed_data
if __name__ == '__main__':
    app.run(debug=False, host = "192.168.0.24")


