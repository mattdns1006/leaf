import os, glob, pdb
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
import matplotlib.pyplot as plt
import argparse

def preprocess(scale=20):
    to_delete = glob.glob("images/*_preprocessed.jpg")
    print("Removing old preprocessed images")
    for f in to_delete:
        os.remove(f)
    img_paths = glob.glob("images/*.jpg")

    print("Get maximum image dimensions in dataset.")
    img_sizes = []
    for img_path in img_paths:
        shape = imread(img_path).shape
        img_sizes.append(shape)
    img_sizes_arr = np.array(img_sizes)
    max_hw = max_h, max_w = img_sizes_arr.max(0)
    out_size = (max_hw/scale).astype(np.int16)
    print("Max height width = {0}. Using scale factor of {1}. Saving files to size {2}.".format(max_hw,scale,out_size))

    img_sizes = []
    for img_path in img_paths:
        img = imread(img_path)
        h,w = img.shape
        title1 = "Image path = {0} has size {1}.".format(img_path,img.shape)
        to_pad_h = int((max_h - h)/2)
        to_pad_w = int((max_w - w)/2)
        img_padded = np.pad(img,((to_pad_h,to_pad_h),(to_pad_w,to_pad_w)),mode='constant')
        resized = imresize(img_padded,out_size,interp='cubic')
        out_path = img_path.replace(".jpg","_preprocessed.jpg")
        imsave(out_path,resized)
    print("Finished!")

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Scale factor for new images.')
    #parser.add_argument('sf', default=15,type=int, help='integer scaling factor')
    #_args = vars(parser.parse_args())
    #preprocess(scale=_args['sf'])
    preprocess(scale=15)
