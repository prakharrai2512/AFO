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
from torch.utils.data import Dataset, DataLoader
from torch import nn

device = "cuda"
class jsrcnn_v2(nn.Module):
    def __init__(self) -> None:
        super(jsrcnn_v2, self).__init__()
        #self.param = param
        self.channel_trans = nn.Sequential(
            nn.Linear(3, 9,bias=False),
            nn.ReLU(),
            nn.Linear(9,128),
            nn.ReLU(),
            nn.Linear(128,3,bias=False),
            nn.Sigmoid(),
        )
        self.maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = self.maskrcnn.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
        self.maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one

    def forward(self, images, targets=None):
        trniml = []
        color_loss = torch.Tensor([0]).to(device)
        for ind in range(len(images)):
            #print(images[ind].shape)
            #img = np.array(img.cpu())
            #images[ind] = images[ind].permute(1,2,0)
            #images[ind].flatten()
            #print(images[ind].shape)
            H = images[ind].shape[1]
            W = images[ind].shape[2]
            kek = images[ind].to(device)
            images[ind] = images[ind].reshape([H*W,3])
            images[ind] = self.channel_trans(images[ind])
            images[ind] = images[ind].reshape([3,H,W])
            
            if self.train:
               sem_val = torch.mul(images[ind],targets[ind]["sem_mask"]).sum((1,2))/targets[ind]["sem_mask_c"]
               #print(targets[ind]["sem_mask"] + targets[ind]["neg_mask"])
               neg_val = torch.mul(images[ind],targets[ind]["neg_mask"]).sum((1,2))/targets[ind]["neg_mask_c"]
               #print("semval",sem_val)
               #print("neg_val",neg_val)
               color_loss = torch.mul(sem_val,neg_val).sum()/(((torch.square(sem_val).sum())**0.5)+((torch.square(neg_val).sum())**0.5))
               #print("top",torch.mul(sem_val,neg_val).sum())
               #print("bot",(((torch.square(sem_val).sum())**0.5)*((torch.square(neg_val).sum())**0.5)+5))
               #color_loss = sem_val
               #color_loss = neg_val - sem_val
            images[ind] = (images[ind] + kek)/2
            #color_loss = (neg_val/sem_val).sum()
               #print(color_loss)
            #images[ind] = images[ind].permute(2,0,1)
            #img = torch.Tensor(img).to(device)
            #trniml.append(img)
        #trniml.to(device)
        ret = self.maskrcnn(images,targets)
        if self.train:
            #print("color_loss",color_loss.shape)
            #color_loss = torch.as_tensor(color_loss,dtype=torch.float16)
            color_loss = color_loss.sum()
            ret["color_loss"] = color_loss
        return ret


model = jsrcnn_v2()
model.to(device)# move model to the right devic


model.load_state_dict(torch.load('jsr_v3.1_2000_0.10339006781578064.torch'))
model.to(device)
model.eval()
model.train = False
torch.no_grad()
for im_name in os.listdir("submission/x/"):
    imgtmp = cv2.imread("submission/x/"+im_name,1).transpose(2,0,1)
    imgtmp = torch.as_tensor(imgtmp, dtype=torch.float32)
    imgtmp = imgtmp/255
    imgtmp = imgtmp.to(device)
    output = model([imgtmp])
    prefix = im_name[0:-4]
    ref = 0
    for i in range(output[0]['masks'].shape[0]):
        print(output[0]['scores'][i])
        if output[0]['scores'][i]<0.5:
            continue
        it = torch.clone(output[0]['masks'][i])
        it = it.detach().to('cpu').numpy()
        it = it.squeeze(0)
        it = (it>=0.5).astype(float)
        it = it*255
        cv2.imwrite("submission/y/"+prefix+"_"+str(i)+".bmp",it)

