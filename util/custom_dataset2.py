from torch.utils.data import Dataset
from PIL import *
from PIL import Image
import os 
import natsort
class CustomDataSet2(Dataset):
    def __init__(self, all_imgs,transform):
        self.transform = transform
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image
    
    def getpath(self,idx):
        img_loc = self.total_imgs[idx]
        return img_loc
