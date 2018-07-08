import os, glob, pdb
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
import matplotlib.pyplot as plt

def preprocess(scale=10,plot=False):
    print("Get maximum image dimensions in dataset.")
    img_paths = glob.glob("images/*.jpg")
    img_sizes = []
    for img_path in img_paths:
        shape = imread(img_path).shape
        img_sizes.append(shape)
    img_sizes_arr = np.array(img_sizes)
    max_hw = max_h, max_w = img_sizes_arr.max(0)
    out_size = (max_hw/scale).astype(np.int16)
    print("Max height width = {0}. Saving files to size {1}.".format(max_hw,out_size))

    img_sizes = []
    for img_path in img_paths[:]:
        img = imread(img_path)
        h,w = img.shape
        title1 = "Image path = {0} has size {1}.".format(img_path,img.shape)
        to_pad_h = int((max_h - h)/2)
        to_pad_w = int((max_w - w)/2)
        img_padded = np.pad(img,((to_pad_h,to_pad_h),(to_pad_w,to_pad_w)),mode='constant')
        resized = imresize(img_padded,out_size)
        out_path = img_path.replace(".jpg","_preprocessed.jpg")
        imsave(out_path,resized)
        if plot == True:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,5))
            title2 = "After padding size = {0}.".format(img_padded.shape)
            ax1.imshow(img,cmap=cm.gray); ax2.imshow(img_padded,cmap=cm.gray)
            ax1.set_title(title1)
            ax2.set_title(title2)
    print("Finished!")

if __name__ == "__main__":
    preprocess()
