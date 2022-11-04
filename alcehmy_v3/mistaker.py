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
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights,MaskRCNN_ResNet50_FPN_Weights
import torch.optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(1, '/home/prakharug/AFO')
sys.path.insert(1, '/home/prakharug/AFO/pycoco')
from pycoco.engine import train_one_epoch, evaluate
import miou_eval as mi


def combine_dims(a, start=0, count=2):
    """ Reshapes numpy array a by combining count dimensions, 
        starting at dimension index start """
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start+count:])


def collate_fn(batch):
    return tuple(zip(*batch))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
print(device)


class MMCellDataset(Dataset):
    def __init__(self,root_dir, tester = False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img = []
        self.nml = []
        self.tester = tester
        self.datano =  root_dir.split('/')[1]
        print(self.datano)
        for im_name in os.listdir(self.root_dir):
            self.nml.append(im_name)
            # tmplist = []
            # lent = len(im_name[:-4])
            # print(im_name,im_name[:lent])
            # imgtmp = cv2.imread(self.root_dir+im_name,1).transpose(2,0,1)
            # tmplist.append(imgtmp[0])
            # tmplist.append(imgtmp[1])
            # tmplist.append(imgtmp[2])
            # for gt_name in os.listdir("../"+self.datano+"/ground_truths"):
            #     if gt_name[0:0+lent]==im_name[:lent] and gt_name[0+lent]=="_":
            #         mask = np.array(cv2.imread("../"+self.datano+"/ground_truths/"+gt_name,0))
            #         tmplist.append(mask)
            # tmplist = np.array(tmplist)
            # #print(tmplist.shape)
            # self.img.append(tmplist)   
    
    def __len__(self):
        return len([name for name in os.listdir(self.root_dir)])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_name = self.nml[idx]
        tmplist = []
        lent = len(im_name[:-4])
        #print(im_name,im_name[:lent])
        imgtmp = cv2.imread(self.root_dir+im_name,1).transpose(2,0,1)
        tmplist.append(imgtmp[0])
        tmplist.append(imgtmp[1])
        tmplist.append(imgtmp[2])
        for gt_name in os.listdir("../"+self.datano+"/ground_truths"):
            if gt_name[0:0+lent]==im_name[:lent] and gt_name[0+lent]=="_":
                mask = np.array(cv2.imread("../"+self.datano+"/ground_truths/"+gt_name,0))
                tmplist.append(mask)
        tmplist = np.array(tmplist)
        #print("Tmplist: ",tmplist.shape)
        patch_stack = tmplist
        patch_stack = np.array(patch_stack)
        # k_w = random.randint(700,patch_stack.shape[2])
        # k_h = random.randint(700,patch_stack.shape[1])
        k_w = patch_stack.shape[2]
        k_h = patch_stack.shape[1]
        o_w = random.randint(0,patch_stack.shape[2]-k_w)
        o_h = random.randint(0,patch_stack.shape[1]-k_h)
        patch_img = patch_stack[0:3,o_h:o_h+k_h,o_w:o_w+k_w]
        patch_img = cv2.cvtColor(patch_img.transpose(1,2,0), cv2.COLOR_BGR2LAB).transpose(2,0,1)
        instances = patch_stack[3:,o_h:o_h+k_h,o_w:o_w+k_w]
        instances = instances.transpose(1,2,0)
        data = {}
        masks = []
        boxes = []
        area = []
        t=0
        for a in range(instances.shape[2]):
            dispim = instances[:,:,a]
            if np.all(dispim == 0):
                continue
            x,y,w,h = cv2.boundingRect(dispim)
            boxes.append([x, y, x+w, y+h])
            area.append(torch.tensor(h*w))
            masks.append(dispim/255)
            t=1
        if t==0:
            #print("Abort")
            if self.tester:
                return "Problem","Hao gai"
            else:
                #print("Abort Abort ")
                return self.__getitem__((idx+1)%len(self.nml))
        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)
        img = torch.as_tensor(patch_img, dtype=torch.float32)
        img = img/255
        data["boxes"] =  boxes
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        data["iscrowd"] = iscrowd
        data["labels"] =  torch.ones((boxes.shape[0],), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        data["area"] = area
        data["image_id"] = torch.tensor(idx)
        return img,data
train = MMCellDataset("../dataset12/images/")
test = MMCellDataset("../dataset12/test/")
fintest = MMCellDataset("../test/images/",True)
train = torch.utils.data.ConcatDataset([train, test])
#print(len(test))
trainloader = DataLoader(train, batch_size=4, shuffle=True,collate_fn = collate_fn,num_workers=10)
#testloader = DataLoader(test, batch_size=1, shuffle=True,collate_fn = collate_fn,num_workers=10)
fintestloader = DataLoader(fintest, batch_size=1, shuffle=True,collate_fn = collate_fn,num_workers=10)



class alchemy(nn.Module):
    def __init__(self,**kwargs) -> None:
        super(alchemy, self).__init__()
        self.vis = False
        self.maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,trainable_backbone_layers=4,**kwargs)
        in_features = self.maskrcnn.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
        self.maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one

    def forward(self, images, targets=None):
        ret = self.maskrcnn(images,targets)
        return ret





def iou(x,y):
    insec = torch.logical_and(x,y)
    uni = torch.logical_or(x,y)
    return torch.sum(insec)/(torch.sum(uni))


def miou_eval(index,gt,ins_pred):
    iouv = 0
    cnt = gt.shape[0]
    mkindex = 0
    for mks in range(0,gt.shape[0]):
        mkindex+=1
        tmpiou = 0
        mk = 0
        for predmks in ins_pred:
            ttoi = iou(gt[mks],predmks)
            if ttoi>tmpiou:
                mk = predmks
            tmpiou = max(tmpiou,ttoi)
            
        #print(tmpiou)
        iouv+=tmpiou
        showgt = np.array([gt[mks].cpu().numpy()]).transpose(1,2,0)*255
           
        showinf = np.array([mk.cpu().numpy()],dtype=np.uint8).transpose(1,2,0)*255
        cv2.imwrite("../comps/"+str(index)+"_"+str(mkindex)+"_"+str(tmpiou.item())[:5]+"_gt.png",showgt)
        cv2.imwrite("../comps/"+str(index)+"_"+str(mkindex)+"_"+str(tmpiou.item())[:5]+"_inf.png",showinf)
        # if(tmpiou.data<0.8):
        #     showgt = np.array([gt[mks].cpu().numpy()]).transpose(1,2,0)*255
           
        #     showinf = np.array([mk.cpu().numpy()],dtype=np.uint8).transpose(1,2,0)*255
        #     #print(showinf)
        #     #showinf = np.array(mk).transpose(1,2,0)
        #     print(showinf.shape)
        #     #print(showinf)
        #     cv2.imshow("gt",showgt)
        #     cv2.imshow("pred",showinf)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
    #print("hello",cnt)
    return iouv,cnt






def evalm(model,fintestloader):
    model.eval()
    iou = 0
    cnt = 0
    for i, data in enumerate(fintestloader, 0):
        images, targets = data
        #print(type(targets))
        if(images[0]=="Problem"):
            continue
        #print(type(images[0]))
        #print(type(images[0]))
        im_writer  = cv2.cvtColor(np.array(images[0].cpu(),dtype=np.float32).transpose(1,2,0),cv2.COLOR_LAB2BGR)
        print(im_writer.shape)
        cv2.imwrite("../comps/"+str(i)+"_img.png",im_writer)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print(model(images)[0]["masks"].squeeze().shape,targets[0]["masks"].shape)
        print(i)
        tiou,tcnt = miou_eval(i,targets[0]["masks"],(model(images)[0]["masks"].squeeze(1)>0.5))
        print(tiou.data/tcnt)
        #print("Txnt: ",tcnt)
        del images,targets
        iou += tiou
        cnt += tcnt
        #print(iou/cnt)
    print("mIoU on test is:",iou.item()/cnt)
    model.train()
    return iou.item()/cnt
    



model = alchemy()
model.to(device)
model.load_state_dict(torch.load("alchemy_0_0.9161_d12.torch"))
evalm(model,fintestloader)