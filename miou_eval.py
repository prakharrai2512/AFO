import numpy as np
import torch

def iou(x,y):
    insec = torch.logical_and(x,y)
    uni = torch.logical_or(x,y)
    return torch.sum(insec)/(torch.sum(uni))


def miou_eval(gt,ins_pred):
    iouv = 0
    cnt = gt.shape[0]
    for mks in range(0,gt.shape[0]):
        tmpiou = 0
        for predmks in ins_pred:
            tmpiou = max(tmpiou,iou(gt[mks],predmks))
        iouv+=tmpiou
        
    #print("hello",cnt)
    return iouv,cnt

