#!/home/bsft21/tinloklee2/miniconda3/envs/depth/bin/python

import numpy as np
from skimage.transform import rescale
import scipy.io as scio
from distortion_model import distortion_model, distortionParameter
import argparse
import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import crop
import random
import cv2 as cv

# For parsing commandline arguments
def argParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sourcedir", type=str, default='./GeoProj/demo')
    parser.add_argument("--datasetdir", type=str, default='./GeoProj/distorted_place365')
    parser.add_argument("--cleardatasetdir", type=bool, default=False)
    args = parser.parse_args()
    return args

def setupDir(args):
    if not os.path.exists(args.sourcedir):
        print(f"Cannot find source directory: {args.sourcedir}")
        return False
    
    #remove two problematic images (if place365 val dataset is used)
    if os.path.exists(os.path.join(args.sourcedir,"Places365_val_00029956.jpg")):
        os.remove(os.path.join(args.sourcedir,"Places365_val_00029956.jpg"))
    if os.path.exists(os.path.join(args.sourcedir,"Places365_val_00031871.jpg")):
        os.remove(os.path.join(args.sourcedir,"Places365_val_00031871.jpg"))

    if args.cleardatasetdir:
        from shutil import rmtree
        rmtree(args.datasetdir)
        print("cleared existing dataset directory")

    if not os.path.exists(args.datasetdir):
        os.mkdir(args.datasetdir)
    for directory in ['train','test']:
        dir_path = os.path.join(args.datasetdir,directory)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for subdirectory in ['image','flow']:
            subdir_path = os.path.join(dir_path,subdirectory)
            if not os.path.exists(subdir_path):
                os.mkdir(subdir_path)

    return True

# def resize_image(original_image,h,w):



def generate_data(cls, k, filename, save_folder_path):
    print(cls,k,save_folder_path.split('/')[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters = distortionParameter(cls)
    
    original_image = cv.imread(filename)
    h,w,c = original_image.shape
    resize_length = min(512,w,h)
    crop_x = random.randint(0,w-resize_length)
    crop_y = random.randint(0,h-resize_length)
    original_image = original_image[crop_y:crop_y+resize_length, crop_x:crop_x+resize_length,:]
    original_image = torch.from_numpy(original_image.transpose((2,0,1))).float() / 255.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_image_tensor = original_image.to(device)
    C, H, W = original_image_tensor.shape
    
    # calculate displacement flow to be applied
    xu, yu = distortion_model(cls, H, W, parameters)
    xu, yu = xu.to(device), yu.to(device)
    
    u = xu - torch.arange(W, device=device).float()
    x_coords = 2*(xu/(W-1))-1
    v = yu - torch.arange(H, device=device).float().unsqueeze(1)
    y_coords = 2*(yu/(H-1))-1

    grid_coords = torch.stack((x_coords,y_coords), dim=-1).unsqueeze(0).to(device)
    distorted_image_tensor = F.grid_sample(original_image_tensor.unsqueeze(0), grid_coords, mode='bilinear', align_corners=True).squeeze(0) # bug may be caused by padding_mode='border'
    
    crop_height = 256
    crop_width = 256
    crop_length = min(128,resize_length-256)
    distorted_image_tensor = crop(distorted_image_tensor,crop_length,crop_length,crop_height,crop_width)

    distorted_image = distorted_image_tensor.permute(1,2,0).cpu().numpy() * 255.0
    distorted_image = np.clip(distorted_image, 0,255).astype(np.uint8)
    crop_u = crop(u.unsqueeze(0),crop_length,crop_length,crop_height,crop_width).squeeze(0).cpu().numpy()
    crop_v = crop(v.unsqueeze(0),crop_length,crop_length,crop_height,crop_width).squeeze(0).cpu().numpy()
    flow_fields = {'u': crop_u,'v': crop_v}

    # print(distorted_image.shape)
    save_image_path = os.path.join(save_folder_path,'image',f'{cls}_{str(k).zfill(6)}.jpg')
    save_mat_path = os.path.join(save_folder_path,'flow',f'{cls}_{str(k).zfill(6)}.mat')
    # save_image_path = '/home/bsft21/tinloklee2/test.jpg'
    # save_mat_path = '/home/bsft21/tinloklee2/test.jpg'
    # save_image_path = f'{cls}_{str(k).zfill(6)}.jpg'
    # save_mat_path = f'{cls}_{str(k).zfill(6)}.mat'
    cv.imwrite(save_image_path, distorted_image)
    scio.savemat(save_mat_path, flow_fields)

def main():
    args = argParser()
    if not setupDir(args):
        exit(1)
    src = f'{args.sourcedir}'
    train_dest = os.path.join(args.datasetdir,"train")
    test_dest = os.path.join(args.datasetdir,"test")

    total = len(os.listdir(args.sourcedir)) # 36498 -> original: 36500
    n_train = int(total*0.8)
    n_test = total-n_train

    for k, filename in enumerate(os.listdir(src)):
        for types in ['barrel','pincushion', 'rotation','shear','projective','wave','none']: 
            if k<n_train:
                generate_data(types, k, os.path.join(src,filename), train_dest)
            else:
                generate_data(types, k, os.path.join(src,filename), test_dest)
    print('FInished dataset generation')

if __name__=='__main__':
    main()
