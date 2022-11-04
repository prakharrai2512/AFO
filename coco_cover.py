import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
from itertools import groupby
import os
import patchify
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image as skimg
import imgaug as ia
import imgaug.augmenters as iaa
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights,MaskRCNN_ResNet50_FPN_Weights
import torch.optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(1, '/home/prakharug/AFO')
sys.path.insert(1, '/home/prakharug/AFO/pycoco')
from pycoco.engine import train_one_epoch, evaluate
import miou_eval as mi
import json
import numpy as np
from pycocotools import mask
from skimage import measure
import help_lo as hlp



def maskers(ground_truth_binary_mask):
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)
    #contour = np.flip(contour, axis=1)
    return ground_truth_area,ground_truth_bounding_box,contours



categories = [
    {'id': 1, 'name': 'cell'},{'id':2,'name':'none'}
]


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle



images = []
for imname in os.listdir('./test/images/'):
    dt  = {}
    dt['file_name'] = imname
    dt['id'] = int(imname.split('.')[0])
    imgt = cv2.imread('./test/images/'+imname)
    dt['height']  = imgt.shape[0]
    dt['width']  = imgt.shape[1]
    images.append(dt)
    #print(imgt.shape)
# for imname in os.listdir('./test/test/'):
#     dt  = {}
#     dt['file_name'] = imname
#     dt['id'] = int(imname.split('.')[0])
#     imgt = cv2.imread('./dataset4/test/'+imname)
#     dt['height']  = imgt.shape[0]
#     dt['width']  = imgt.shape[1]
#     images.append(dt)

print(len(images))

annotations = []
annoid = 1
bt = 0
for imname in os.listdir('./test/ground_truths/'):
    #print(bt)
    bt+=1
    dt = {}
    tll = []
    with np.printoptions(threshold=np.inf):
        #print(cv2.imread('./dataset12/ground_truths/'+imname,0))/255
        pass
    maskt = np.array(cv2.imread('./test/ground_truths/'+imname,0),dtype = np.uint8).squeeze()
    #print(maskt.shape)
    box2 = cv2.boundingRect(maskt)
    area,bbox,poly = maskers(maskt)
    # segrl= 0
    # segr = []
    for contour in poly:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        if(len(segmentation)>20):
            segrl = len(segmentation)
            segr = segmentation
            tll.append(segmentation)
        
    # if(len(dt["segmentation"])>1):
    #     print("arre bt")
    #print(poly)
    # polt = []
    # for ar in poly:
    #     polt.append(ar.flatten())
    # polt = [polte.tolist() for polte in polt]
    bbox = [int(btbox) for btbox in bbox]
    
    #print(polt)
    #print(bbox,area)
    #dt['area'] = 
    #print(poly)
    name = imname.split('.')[0]
    fname = int(name.split('_')[0])
    lname = int(name.split('_')[1])
    dt["id"] = annoid
    annoid+=1
    dt["image_id"] = fname
    dt["category_id"] = 1
    #dt["segmentation"] = binary_mask_to_rle(maskt)
    dt["segmentation"] = hlp.binary_mask_to_polygon(maskt)
    dt["area"] = int(area)
    dt["bbox"] = bbox
    dt["iscrowd"] = 0
    if(len(dt["segmentation"])>0):
        annotations.append(dt)
    
print(len(annotations))

info =  {
        "description": "SegPC-Custom COCO",
        "version": "1.0",
        "year": 2022,
        "contributor": "Prakhar Rai",
        "date_created": "2022/10/25"
    }

    
dicty = {"info" : info, "images" : images,"annotations" : annotations,"categories" : categories} 

import json
with open('result_test.json', 'w') as fp:
    json.dump(dicty, fp)