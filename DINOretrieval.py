"""
Author: Adriel Kuek
Date Created: 18 Oct 2021
Version: 0.1
Email: adrielkuek@gmail.com
Status: Devlopment

Description:
Image retrieval through DINO representations (Self-Distillation with no Labels)

"""
import os
import sys
import argparse

import torch
from torch import nn
# import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from custom_dataset2 import CustomDataSet2

from util import utils
from util import vision_transformer as vits

class DINO(object):
    def __init__(self, model_dir, data_loader, dataset_features):
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load DINO ViTsmall/8x8-small model - Reported 78.3% Top-1 accuracy on ImageNet with kNN Classification
        print(f'LOADING DINO PRETRAINED . . .')
        self.model = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
        print(f"Model ViT-8x8 built.")

        # Load model to CUDA
        self.model.to(self.device)
        utils.load_pretrained_weights(self.model, model_dir, checkpoint_key="teacher", 
                                        model_name="vit_small", patch_size=8)
        
        # Set model to inference
        self.model.eval()

        # Load dataset loader
        self.dataset_loader = torch.load(data_loader)

        # Load dataset features
        self.dataset_features = torch.load(dataset_features)

    def kNN_retrieval(self, input_img, Topk):

        # ============ preparing data ... ============
        # IMG TRANSFORMATIONS
        transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset_query = CustomDataSet2([input_img], transform=transform)
        
        # Insert into Dataloader
        data_loader_query = torch.utils.data.DataLoader(
            dataset_query,
            batch_size=128,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        # Extract Features
        print("Extracting features for query image")
        with torch.no_grad():
            query_features = self.extractFeatures(data_loader_query, 
                                                use_cuda=True, multiscale=False, 
                                                batch_size_per_gpu=128)
                                                
        # Normalise Features
        if utils.get_rank() == 0:
            query_features = nn.functional.normalize(query_features, dim=1, p=2)
            query_features = query_features.cpu()

        # Perform kNN retrieval
        image_list = []

        # Transpose for computation
        dataset_features = self.dataset_features.t()
        similarity = torch.mm(query_features, dataset_features)

        # Distances -> 1: Similar, 0: Dissimilar
        distances, indices = similarity.topk(Topk, largest=True, sorted=True)

        for i in range(Topk):
            img_name = os.path.basename(self.dataset_loader.dataset.getpath(indices[0][i]))
            image_list.append(img_name)
        
        return image_list

    # @torch.no_grad()
    def extractFeatures(self, data_loader, use_cuda=True, multiscale=False, 
                            batch_size_per_gpu = 128, imagenet = 1):
        metric_logger = utils.MetricLogger(delimiter="  ")
        features = None

        z = 0
        length = len(data_loader.dataset) #800
        num_iter = length//batch_size_per_gpu #6 but supposedly 6.25 so u need 7 iter, where last iter idx = 6
        last_iter = num_iter  #6
        last_iter_len = length - last_iter * batch_size_per_gpu #800-6*128 = 32

        # what if you have 768 instead.
        # num_iter = 6 last iter idx shld be 5.

        for samples in metric_logger.log_every(data_loader, 10):
            if z!= last_iter:
                samples = samples.cuda(non_blocking=True)
                if multiscale:
                    feats = utils.multi_scale(samples, self.model)
                else:
                    # FEATURES EXTRACTED HERE, USING MODEL
                    feats = self.model(samples).clone()

                # init storage feature matrix
                if features is None:
                    # INTIALIZE FEATURE_MATRIX WITH ZEROS
                    features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
                    if use_cuda:
                        features = features.cuda(non_blocking=True)
                    print(f"Storing features into tensor of shape {features.shape}")

                start = z*batch_size_per_gpu #6*128 = 768
                end = start + batch_size_per_gpu
                index_all = torch.tensor(range(start,end)).cuda()

                # share features between processes
                # feats_all = torch.empty(
                #     1,
                #     feats.size(0),
                #     feats.size(1),
                #     dtype=feats.dtype,
                #     device=feats.device,
                # )
                # ADD INTO FEATURE_MATRIX
                output_l = list([feats])

                # update storage feature matrix
                if use_cuda:
                    features.index_copy_(0, index_all, torch.cat(output_l))
                else:
                    features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
            else:
                samples = samples.cuda(non_blocking=True)
                if multiscale:
                    feats = utils.multi_scale(samples, self.model)
                else:
                    # FEATURES EXTRACTED HERE, USING MODEL
                    feats = self.model(samples).clone()

                # init storage feature matrix
                if features is None:
                    # INTIALIZE FEATURE_MATRIX WITH ZEROS
                    features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
                    if use_cuda:
                        features = features.cuda(non_blocking=True)
                    print(f"Storing features into tensor of shape {features.shape}")

                start = z*batch_size_per_gpu
                end = start + last_iter_len
                index_all = torch.tensor(range(start,end)).cuda()

                # share features between processes
                # feats_all = torch.empty(
                #     1,
                #     feats.size(0),
                #     feats.size(1),
                #     dtype=feats.dtype,
                #     device=feats.device,
                # )
                # ADD INTO FEATURE_MATRIX
                output_l = list([feats])

                # update storage feature matrix
                if use_cuda:
                    features.index_copy_(0, index_all, torch.cat(output_l))
                else:
                    features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
            z += 1

        return features

# For Testing of class object
def main():

    model_dir = '/home/user/Adriel/MTech-IS/GradCert_PatternRecognition/PRS_PractiseModule/Imantics.io/dino_deitsmall8_pretrain.pth'
    data_loader = '/home/user/Adriel/MTech-IS/GradCert_PatternRecognition/PRS_PractiseModule/Imantics.io/dino_features/Ourdataset_loader.pth'
    dataset_features = '/home/user/Adriel/MTech-IS/GradCert_PatternRecognition/PRS_PractiseModule/Imantics.io/dino_features/Ourdataset_features.pth'
    
    imageResults_list = []

    dino = DINO(model_dir, data_loader, dataset_features)

    # Test with sample image
    # Retrieve Test Image Query
    imgSample_filepath = '/media/user/New Volume/TINKERMAN/OurDataset/PRS_CombinedDataset2/train/ODS00020.jpg'

    imageResults_list = dino.kNN_retrieval(imgSample_filepath, 15)
    print(imageResults_list)

if __name__ == "__main__":
    main()


