import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def display_dataset(images, rowscols=(2,10)) :
    print("Number of images : ", len(images))
    sample_img = mpimg.imread(images[0])
    print("Image shape : ", sample_img.shape)
    print("Image min : ", sample_img.min())
    print("Image max : ", sample_img.max())
    print("Image data type : ", sample_img.dtype)
    nrows,ncols = rowscols
    num_img = nrows*ncols
    images_select = random.sample(images, num_img)
    fig, ax = plt.subplots(nrows,ncols, figsize=(2*ncols,2*nrows))
    for img,ax in zip(images_select, ax.flatten()) :
        img = mpimg.imread(img)
        ax.imshow(img)
        ax.axis('off')
    plt.show()
