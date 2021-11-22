#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from parhugin import multiFunc
    parhugin_installed = True
except ImportError:
    print("[WARNING] parhugin (https://github.com/kasra-hosseini/parhugin) is not installed, continue without it.")
    parhugin_installed = False

class patchTorchDataset(Dataset):
    
    def __init__(self, 
                 patchframe, 
                 transform=None, 
                 label_col="label", 
                 convert2="RGB", 
                 input_col=0):
        """Instantiate patchTorchDataset and collect some info

        Parameters
        ----------
        patchframe : DataFrame
            DataFrame that contains patch paths/labels
        transform : torchvision transforms, optional
            torchvision transforms, by default None
        label_col : str, optional
            name of the column that contains labels, by default "label"
        """
        self.patchframe = patchframe
        self.label_col = label_col
        self.convert2 = convert2
        self.input_col = input_col

        if self.label_col in self.patchframe.columns.tolist():
            self.uniq_labels = self.patchframe[self.label_col].unique().tolist()
        else:
            self.uniq_labels = "NS"
            
        if transform in ["train", "val"]:
            self.transform = self._default_transform(transform)
        elif transform is None:
            raise ValueError(f"transform argument is not set.")
        else:
            self.transform = transform
        
    def __len__(self):
        return len(self.patchframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])
        image = Image.open(img_path).convert(self.convert2)

        image = self.transform(image)
        
        if self.label_col in self.patchframe.iloc[idx].keys():
            image_label = self.patchframe.iloc[idx][self.label_col]
        else:
            image_label = -1

        return image, image_label 

    def return_orig_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])
        image = Image.open(img_path).convert(self.convert2)
        return image
    
    def _default_transform(self, t_type: str = "train", resize2: int = 224):
        """Default transformations

        Parameters
        ----------
        t_type : str, optional
            transformation type. This can be "train"/"val", by default "train"
        """
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        data_transforms = {
            'train': transforms.Compose(
                [transforms.Resize(resize2),
                 transforms.RandomApply([
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomVerticalFlip(),
                     #transforms.ColorJitter(brightness=0.3, contrast=0.3),
                     ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
                ]),
            'val': transforms.Compose(
                [transforms.Resize(resize2),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
                ]),
        }
        return data_transforms[t_type]

# --- Dataset that returns an image, its context and its label 
class patchContextDataset(Dataset):
    
    def __init__(self, 
                 patchframe,
                 transform1=None, 
                 transform2=None, 
                 label_col="label", 
                 convert2="RGB", 
                 input_col=0, 
                 context_save_path="./maps/maps_context", 
                 create_context=False,
                 par_path="./maps",
                 x_offset=1.,
                 y_offset=1.,
                 slice_method="scale"):
        """Instantiate patchContextDataset and collect some info

        Parameters
        ----------
        patchframe : DataFrame
            DataFrame that contains patch paths/labels
        transform : torchvision transforms, optional
            torchvision transforms, by default None
        label_col : str, optional
            name of the column that contains labels, by default "label"
        """
        self.patchframe = patchframe
        self.label_col = label_col
        self.convert2 = convert2
        self.input_col = input_col
        self.par_path = par_path
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.slice_method = slice_method
        self.create_context = create_context

        if not self.create_context:
            self.context_save_path = os.path.abspath(context_save_path)

        if self.label_col in self.patchframe.columns.tolist():
            self.uniq_labels = self.patchframe[self.label_col].unique().tolist()
        else:
            self.uniq_labels = "NS"
            
        if transform1 in ["train", "val"]:
            self.transform1 = self._default_transform(transform1)
        elif transform1 is None:
            raise ValueError(f"transform argument is not set.")
        else:
            self.transform1 = transform1

        if transform2 in ["train", "val"]:
            self.transform2 = self._default_transform(transform2)
        elif transform2 is None:
            raise ValueError(f"transform argument is not set.")
        else:
            self.transform2 = transform2
    
    def save_parents(self, num_req_p=10, sleep_time=0.001, use_parhugin=True, par_split="#", loc_split="-", overwrite=False):
        
        if parhugin_installed and use_parhugin:
            myproc = multiFunc(num_req_p=num_req_p, sleep_time=sleep_time)
            list_jobs = []
            for idx in range(len(self.patchframe)):
                list_jobs.append([self.save_parents_idx, (idx, par_split, loc_split, overwrite)])

            print(f"Total number of jobs: {len(list_jobs)}")
            # and then adding them to myproc
            myproc.add_list_jobs(list_jobs)
            myproc.run_jobs()
        else:
            for idx in range(len(self.patchframe)):
                self.save_parents_idx(idx)
            
    def save_parents_idx(self, idx, par_split="#", loc_split="-", overwrite=False, return_image=False):

        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])
        img_rd = Image.open(img_path).convert(self.convert2)

        if not return_image:
            os.makedirs(self.context_save_path, exist_ok=True)

            path2save_context = os.path.join(self.context_save_path, os.path.basename(img_path))

            if os.path.isfile(path2save_context) and (not overwrite):
                return
         
        if self.slice_method in ["scale"]:
            # size: (width, height)         
            tar_y_offset = int(img_rd.size[1]*self.y_offset)
            tar_x_offset = int(img_rd.size[0]*self.x_offset)
        else:
            tar_y_offset = self.y_offset
            tar_x_offset = self.x_offset

        par_name = os.path.basename(img_path).split(par_split)[1]
        split_path = os.path.basename(img_path).split(loc_split)
        min_x, min_y, max_x, max_y = int(split_path[1]), int(split_path[2]), int(split_path[3]), int(split_path[4])

        if self.par_path in ["dynamic"]:
            par_path2read = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(img_path))), par_name)
        else:
            par_path2read = os.path.join(os.path.abspath(self.par_path), par_name)
        
        par_img = Image.open(par_path2read).convert(self.convert2)

        min_y_par = max(0, min_y-tar_y_offset)
        min_x_par = max(0, min_x-tar_x_offset)
        max_x_par = min(max_x+tar_x_offset, np.shape(par_img)[1])
        max_y_par = min(max_y+tar_y_offset, np.shape(par_img)[0])

        pad_activate = False
        top_pad = left_pad = right_pad = bottom_pad = 0
        if (min_y-tar_y_offset) < 0:
            top_pad = abs(min_y-tar_y_offset)
            pad_activate = True
        if (min_x-tar_x_offset) < 0:
            left_pad = abs(min_x-tar_x_offset)
            pad_activate = True
        if (max_x+tar_x_offset) > np.shape(par_img)[1]:
            right_pad = max_x + tar_x_offset - np.shape(par_img)[1]
            pad_activate = True
        if (max_y+tar_y_offset) > np.shape(par_img)[0]:
            bottom_pad = max_y + tar_y_offset - np.shape(par_img)[0]
            pad_activate = True

        #par_img = par_img[min_y_par:max_y_par, min_x_par:max_x_par]
        par_img = par_img.crop((min_x_par, min_y_par, max_x_par, max_y_par))

        if pad_activate:
            padding = (left_pad, top_pad, right_pad, bottom_pad)
            par_img = ImageOps.expand(par_img, padding)

        if return_image:
            return par_img
        elif not os.path.isfile(path2save_context):
            par_img.save(path2save_context)
    
    def __len__(self):
        return len(self.patchframe)
    
    def return_orig_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])
        image = Image.open(img_path).convert(self.convert2)
        return image
    
    def plot_sample(self, indx):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(transforms.ToPILImage()(self.__getitem__(indx)[0]))
        plt.title("Patch", size=18)
        plt.xticks([]); plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(transforms.ToPILImage()(self.__getitem__(indx)[1]))
        plt.title("Context", size=18)
        plt.xticks([]); plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.show()
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])
        image = Image.open(img_path).convert(self.convert2)

        if self.create_context:
            context_img = self.save_parents_idx(idx, return_image=True)
        else:
            context_img = Image.open(os.path.join(self.context_save_path, os.path.basename(img_path))).convert(self.convert2)
        
        image = self.transform1(image)
        context_img = self.transform2(context_img)
        
        if self.label_col in self.patchframe.iloc[idx].keys():
            image_label = self.patchframe.iloc[idx][self.label_col]
        else:
            image_label = -1

        return image, context_img, image_label
