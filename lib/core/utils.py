'''
Some useful tools in training process
'''
import torch
import cv2
import numpy as np
import math
import torch.nn.functional as F
import torchvision.transforms as T
from tsnecuda import TSNE
from collections import OrderedDict
import matplotlib.pyplot as plt
class AverageMeter(object):
    """
    Computes and store the average the current value
    """
    def __init__(self):
        self.val = 0  # current value 
        self.avg = 0  # avage value
        self.sum = 0  
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def tsne_vis(feature, feature_labels, vis_path):
    feature_np = feature.detach().cpu().numpy()
    feature_embeddings = TSNE().fit_transform(feature_np)
    vis_x = feature_embeddings[:, 0]
    vis_y = feature_embeddings[:, 1]
    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(vis_x, vis_y, c=feature_labels, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig(vis_path)


