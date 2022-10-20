import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
import patchify
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image as skimg
import imgaug as ia
import imgaug.augmenters as iaa
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
import torch.optim

def combine_dims(a, start=0, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])


instance_toimage = []

for im_name in os.listdir("./dataset/images"):
    tmplist = []
    lent = len(im_name[:-4])
    print(im_name,im_name[:lent])
    imgtmp = cv2.imread("./dataset/images/"+im_name,1).transpose(2,0,1)
    #print(imgtmp[0].shape)
    tmplist.append(imgtmp[0])
    tmplist.append(imgtmp[1])
    tmplist.append(imgtmp[2])
    for gt_name in os.listdir("./dataset/ground_truths"):
        #print(gt_name[10+lent],gt_name[10:10+lent])
        if gt_name[10:10+lent]==im_name[:lent] and gt_name[10+lent]=="_":
            mask = np.array(cv2.imread("./dataset/ground_truths/"+gt_name,0))
            # mask = (mask > 0).astype(np.uint8) 
            tmplist.append(mask)
    tmplist = np.array(tmplist)
    print(tmplist.shape)
    instance_toimage.append(tmplist)

for im_name in os.listdir("./dataset/test"):
    tmplist = []
    lent = len(im_name[:-4])
    print(im_name,im_name[:lent])
    imgtmp = cv2.imread("./dataset/test/"+im_name,1).transpose(2,0,1)
    #print(imgtmp[0].shape)
    tmplist.append(imgtmp[0])
    tmplist.append(imgtmp[1])
    tmplist.append(imgtmp[2])
    for gt_name in os.listdir("./dataset/ground_truths"):
        #print(gt_name[10+lent],gt_name[10:10+lent])
        if gt_name[10:10+lent]==im_name[:lent] and gt_name[10+lent]=="_":
            mask = np.array(cv2.imread("./dataset/ground_truths/"+gt_name,0))
            # mask = (mask > 0).astype(np.uint8) 
            tmplist.append(mask)
    tmplist = np.array(tmplist)
    print(tmplist.shape)
    instance_toimage.append(tmplist)


batchSize = 2
img_siz = [512,512]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
print(device)

from readline import append_history_file
from unittest.mock import patch


def loadBatch():
    rtrnd=random.randint(0,4)
    seq = iaa.Sequential([
    #iaa.Fliplr(0.5), # horizontal flips
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.Rot90((rtrnd), keep_size=False)
    ])
    batch_Imgs=[]
    batch_Data=[]
    for i in range(batchSize):
        idx=random.randint(0,len(instance_toimage)-1)
        #print("instance_toimage[idx].shape : ",instance_toimage[idx].shape)
        patch_stack = seq(images=instance_toimage[idx])
        #patch_stack = instance_toimage[idx]
        patch_stack = np.array(patch_stack)
        k_w = random.randint(512,patch_stack.shape[2])
        k_h = random.randint(512,patch_stack.shape[1])
        o_w = random.randint(0,patch_stack.shape[2]-k_w)
        o_h = random.randint(0,patch_stack.shape[1]-k_h)
        patch_img = patch_stack[0:3,o_h:o_h+k_h,o_w:o_w+k_w]
        instances = patch_stack[3:,o_h:o_h+k_h,o_w:o_w+k_w]
        #print("patch_img.shape",patch_img.shape)
        #print(patch_img.shape)
        #patch_img = patch_img.transpose(1,2,0)
        instances = instances.transpose(1,2,0)
        #print(instances.shape)
        # cv2.imshow("jsr",patch_img)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
        data = {}
        masks = []
        boxes = []
        t=0
        for a in range(instances.shape[2]):
            #dispim = np.expand_dims(instances[:,:,a], axis=2)
            dispim = instances[:,:,a]
            if np.all(dispim == 0):
                continue
            x,y,w,h = cv2.boundingRect(dispim)
            boxes.append([x, y, x+w, y+h])
            #dispim = dispim.transpose(2,0,1)
            #print(dispim.shape)
            masks.append(dispim/255)
            t=1
        #     cv2.imshow("jsr1",dispim)
        #     cv2.waitKey(0)  
        # cv2.destroyAllWindows() 
        masks = np.array(masks)
        
        if t==0:
            return loadBatch()
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        img = torch.as_tensor(patch_img, dtype=torch.float32)
        img = img/255
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((boxes.shape[0],), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data)
    #batch_Imgs = torch.Tensor(batch_Imgs)
    #batch_Imgs.to("cuda")
    #batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    #print("Load")
    #for i in range(len(batch_Imgs)):
    #    print(batch_Imgs[i].shape)
    #    print(batch_Data[i]["masks"].shape)
    return batch_Imgs, batch_Data
#loadBatch()

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)  # load an instance segmentation model pre-trained pre-trained on COCO
in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
model.to(device)# move model to the right devic



optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
model.train()
lmbda = lambda epoch: 0.8
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
for i in range(2000):
            images, targets = loadBatch()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            # for losst in loss_dict.keys():
            #     print(losst,loss_dict[losst])
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()
            print(i,'loss:', losses.item())
            if i%200==0:
                #optimizer = torch.optim.AdamW(params=model.parameters(), lr=optimizer.)
                scheduler.step()
                torch.save(model.state_dict(), "cif"+str(i)+".torch")
