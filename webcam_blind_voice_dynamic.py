import numpy as np
import os
import urllib
import urllib.request as allib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import pytesseract
import engineio

import torch
from torch.autograd import Variable as V
import models as models
from torchvision import transforms as trn
from torch.nn import functional as F

import pyttsx3
engine = pyttsx3.init()

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFont

arch = 'resnet18'
model_file = 'whole_%s_places365_python36.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'  

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

base_path = 'D:\\Semester 6\\CVIP Lab\\Project\\Blind-Assistance-Object-Detection'
data_path = 'models\\research\\object_detection\\data'
file_name = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = os.path.join(base_path, data_path, file_name)

NUM_CLASSES = 90

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    print ('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
    print ('Download complete')
else:
    print ('Model already exists')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

url = 'http://10.67.208.240:8080//shot.jpg'

import cv2
cap = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        ret = True
        while (ret):
            ret,image_np = cap.read()

            if cv2.waitKey(20) & 0xFF == ord('b'): 
                cv2.imwrite('opencv'+'.jpg', image_np) 

                # Object detection using Places365 model
                model_file = 'whole_%s_places365_python36.pth.tar' % arch
                if not os.access(model_file, os.W_OK):
                    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
                    os.system('wget ' + weight_url)

                useGPU = 1
                if useGPU == 1:
                    model = torch.load(model_file)
                else:
                    model = torch.load(model_file, map_location=lambda storage, loc: storage)

                model.eval()
                centre_crop = trn.Compose([
                    trn.Resize((256,256)),
                    trn.CenterCrop(224),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                file_name = 'categories_places365.txt'
                if not os.access(file_name, os.W_OK):
                    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
                    os.system('wget ' + synset_url)
                classes = list()
                with open(file_name) as class_file:
                    for line in class_file:
                        classes.append(line.strip().split(' ')[0][3:])
                classes = tuple(classes)

                img_name = 'opencv.jpg'
                if not os.access(img_name, os.W_OK):
                    img_url = 'http://places.csail.mit.edu/demo/' + img_name
                    os.system('wget ' + img_url)

                img = Image.open(img_name)
                input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

                logit = model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)

                print('POSSIBLE SCENES ARE: ' + img_name)
                engine.say("Possible Scene may be")
                engine.say(img_name)

                for i in range(0, 5):
                    engine.say(classes[idx[i]])
                    print('{}'.format(classes[idx[i]]))
                                                                #batch size , height, width,channel
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.array(np.squeeze(classes), dtype=int),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            for i, b in enumerate(boxes[0]):
                if scores[0][i] > 0.5:
                    # Get the label and distance of the detected object
                    label = category_index[classes[0][i]]['name']
                    score = scores[0][i]
                    ymin, xmin, ymax, xmax = boxes[0][i]

                    # Calculate the midpoint of the bounding box
                    mid_x = (xmin + xmax) / 2
                    mid_y = (ymin + ymax) / 2

                    # Calculate approximate distance
                    apx_distance = round(((1 - (xmax - xmin)) ** 4), 1)

                    # Add text to the image
                    cv2.putText(image_np, '{}: {}'.format(label, apx_distance), (int(mid_x * 800), int(mid_y * 450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Determine color for the bounding box based on distance
                    color = (0, 255, 0)  # Default: Green
                    if apx_distance < 0.5:
                        color = (0, 0, 255)  # Red for too close

                    # Draw bounding box
                    cv2.rectangle(image_np, (int(xmin * 800), int(ymin * 450)), (int(xmax * 800), int(ymax * 450)), color, 2)

                    # Speak out the label and distance
                    engine.say("{} is at a distance of {} units".format(label, apx_distance))
                    engine.runAndWait()

                    # Check if object is too close and give a warning
                    if apx_distance < 0.4:
                        engine.say("Warning! {} is too close".format(label))
                        engine.runAndWait()

            # Show the image with detections
            cv2.imshow('image', cv2.resize(image_np, (1024, 768)))
            if cv2.waitKey(2) & 0xFF == ord('t'):
                cv2.destroyAllWindows()
                cap.release()
                break

#{1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'bicycle'}, 3: {'id': 3, 'name': 'car'}, 4: {'id': 4, 'name': 'motorcycle'}, 5: {'id': 5, 'name': 'airplane'}, 6: {'id': 6, 'name': 'bus'}, 7: {'id': 7, 'name': 'train'}, 8: {'id': 8, 'name': 'truck'}, 9: {'id': 9, 'name': 'boat'}, 10: {'id': 10, 'name': 'traffic light'}, 11: {'id': 11, 'name': 'fire hydrant'}, 13: {'id': 13, 'name': 'stop sign'}, 14: {'id': 14, 'name': 'parking meter'}, 15: {'id': 15, 'name': 'bench'}, 16: {'id': 16, 'name': 'bird'}, 17: {'id': 17, 'name': 'cat'}, 18: {'id': 18, 'name': 'dog'}, 19: {'id': 19, 'name': 'horse'}, 20: {'id': 20, 'name': 'sheep'}, 21: {'id': 21, 'name': 'cow'}, 22: {'id': 22, 'name': 'elephant'}, 23: {'id': 23, 'name': 'bear'}, 24: {'id': 24, 'name': 'zebra'}, 25: {'id': 25, 'name': 'giraffe'}, 27: {'id': 27, 'name': 'backpack'}, 28: {'id': 28, 'name': 'umbrella'}, 31: {'id': 31, 'name': 'handbag'}, 32: {'id': 32, 'name': 'tie'}, 33: {'id': 33, 'name': 'suitcase'}, 34: {'id': 34, 'name': 'frisbee'}, 35: {'id': 35, 'name': 'skis'}, 36: {'id': 36, 'name': 'snowboard'}, 37: {'id': 37, 'name': 'sports ball'}, 38: {'id': 38, 'name': 'kite'}, 39: {'id': 39, 'name': 'baseball bat'}, 40: {'id': 40, 'name': 'baseball glove'}, 41: {'id': 41, 'name': 'skateboard'}, 42: {'id': 42, 'name': 'surfboard'}, 43: {'id': 43, 'name': 'tennis racket'}, 44: {'id': 44, 'name': 'bottle'}, 46: {'id': 46, 'name': 'wine glass'}, 47: {'id': 47, 'name': 'cup'}, 48: {'id': 48, 'name': 'fork'}, 49: {'id': 49, 'name': 'knife'}, 50: {'id': 50, 'name': 'spoon'}, 51: {'id': 51, 'name': 'bowl'}, 52: {'id': 52, 'name': 'banana'}, 53: {'id': 53, 'name': 'apple'}, 54: {'id': 54, 'name': 'sandwich'}, 55: {'id': 55, 'name': 'orange'}, 56: {'id': 56, 'name': 'broccoli'}, 57: {'id': 57, 'name': 'carrot'}, 58: {'id': 58, 'name': 'hot dog'}, 59: {'id': 59, 'name': 'pizza'}, 60: {'id': 60, 'name': 'donut'}, 61: {'id': 61, 'name': 'cake'}, 62: {'id': 62, 'name': 'chair'}, 63: {'id': 63, 'name': 'couch'}, 64: {'id': 64, 'name': 'potted plant'}, 65: {'id': 65, 'name': 'bed'}, 67: {'id': 67, 'name': 'dining table'}, 70: {'id': 70, 'name': 'toilet'}, 72: {'id': 72, 'name': 'tv'}, 73: {'id': 73, 'name': 'laptop'}, 74: {'id': 74, 'name': 'mouse'}, 75: {'id': 75, 'name': 'remote'}, 76: {'id': 76, 'name': 'keyboard'}, 77: {'id': 77, 'name': 'cell phone'}, 78: {'id': 78, 'name': 'microwave'}, 79: {'id': 79, 'name': 'oven'}, 80: {'id': 80, 'name': 'toaster'}, 81: {'id': 81, 'name': 'sink'}, 82: {'id': 82, 'name': 'refrigerator'}, 84: {'id': 84, 'name': 'book'}, 85: {'id': 85, 'name': 'clock'}, 86: {'id': 86, 'name': 'vase'}, 87: {'id': 87, 'name': 'scissors'}, 88: {'id': 88, 'name': 'teddy bear'}, 89: {'id': 89, 'name': 'hair drier'}, 90: {'id': 90, 'name': 'toothbrush'}}


