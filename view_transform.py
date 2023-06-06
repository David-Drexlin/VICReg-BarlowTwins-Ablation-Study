# view_transform.py
from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


### Copied from https://github.com/facebookresearch/vicreg/blob/main/augmentations.py
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
###


class ViewTransform(object):
    def __init__(self, num):
        self.num = num
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            GaussianBlur(p=0.5),
            Solarization(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomRotation(180),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],p=0.8),            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])

    def __call__(self, x):
        # add instances / stack for proposed research question
        transforms = []
        for i in range(self.num):
            transforms.append(self.transform(x))
        return transforms


"""
#Barlow Twins Paper - with modifications

def barlow_twins(Z):
    la = 0.005 
    
    #input is [batch_size, 1000]
    #conv1d requires 3 dimensions, target CC is DxD i.e. 1000x1000

    N = Z[0].shape[0]
    D = Z[0].shape[1]
    
    loss = 0

    for i in range(len(Z)): 
        for j in range(len(Z)): 
            zi = Z[i] - Z[i].mean(dim=0)
            zj = Z[j] - Z[j].mean(dim=0)

            c = torch.matmul(zi.T, zj)
            c_diff = (c - torch.eye(D)).pow(2)
            
            off_diags = (torch.ones(c_diff.shape).fill_diagonal_(0))*la
            c_diff = c_diff*off_diags

            loss += c_diff.sum()
    
    return loss / len(Z)**2
"""