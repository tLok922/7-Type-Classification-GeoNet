import numpy as np
import scipy.io as spio
import cv2
import argparse
import os
def argParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetdir", type=str, default='/home/bsft21/tinloklee2/GeoProj-master/GeoProj/distorted_place365/test/flow')
    parser.add_argument("--savedir", type=str, default='/home/bsft21/tinloklee2/GeoProj-master/GeoProj/visualised')
    args = parser.parse_args()
    return args

def setupDir(args):
    if not os.path.exists(args.datasetdir):
        print(f"Cannot find dataset directory: {args.datasetdir}")
        return False

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    return True

def load_distortion_flow(mat_file):
    data = spio.loadmat(mat_file)
    u = data.get('u', None)
    v = data.get('v', None)
    flow = np.stack((u,v), axis=-1)
    return flow

def visualise_flow(flow, output_image):
# adapted from https://github.com/sahakorn/Python-optical-flow-tracking/blob/master/optical_flow.py
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    v = np.sqrt(fx*fx+fy*fy)
    angle = np.arctan2(fy, fx) + np.pi
    magnitude = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = angle*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    visualisation = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imwrite(output_image, visualisation)
    print(f"Saved path: {output_image}")

def main():
    args = argParser()
    if not setupDir(args):
        exit(1)
    for mat_file in os.listdir(args.datasetdir):
        # mat_file = "/public/tinloklee2/distorted_place365/test/flow/shear_036497.mat"
        mat_file = os.path.join(args.datasetdir,mat_file)
        output_path = os.path.join(args.savedir,mat_file.split('/')[-1].replace(".mat",'.png'))

        distortion_flow = load_distortion_flow(mat_file)
        visualise_flow(distortion_flow, output_path)

if __name__=='__main__':
    main()
