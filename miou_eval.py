import numpy as np
import torch
import cv2

def iou(x,y):
    insec = torch.logical_and(x,y)
    uni = torch.logical_or(x,y)
    return torch.sum(insec)/(torch.sum(uni))


def miou_eval(gt,ins_pred):
    iouv = 0
    cnt = gt.shape[0]
    for mks in range(0,gt.shape[0]):
        tmpiou = 0
        mk = 0
        for predmks in ins_pred:
            ttoi = iou(gt[mks],predmks)
            tmpiou = max(tmpiou,ttoi)
            if ttoi>tmpiou:
                mk = predmks
        #print(tmpiou)
        iouv+=tmpiou
        # if(tmpiou.data<0.8):
        #     showgt = np.array([gt[mks].cpu().numpy()]).transpose(1,2,0)
        #     showinf = np.array(mk).transpose(1,2,0)
        #     cv2.imshow("gt",showgt)
        #     cv2.imshow("pred",showinf)
    #print("hello",cnt)
    return iouv,cnt

