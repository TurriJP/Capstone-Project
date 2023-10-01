import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

import maskslic as seg
import skimage2.segmentation as seg2

#vv_src = rasterio.open('data/VV_VH.tif')
#vv = vv_src.read()
vv = imread('data/VV_VH.tif')
# vv = vv[:3, :, :]
# vv = vv.transpose(1, 2, 0)

# image = vv[0:300][0:300][:][:]
# print(len(image[0]))

image = vv[0:300]#, 0:300]

vv_slic = seg.slic(image)
# plt.imshow(vv_slic)
plt.imsave('result.png', vv_slic)

image_with_bounds = seg2.mark_boundaries(image=image/255, label_img=vv_slic)
plt.imsave('bounded_result.png', image_with_bounds)

def rgb():
    vv_src = rasterio.open('data/imagem_RGB.png')
    vv = vv_src.read()
    # vv = imread('data/imagem_RGB.png')
    vv = vv[:3, :, :]
    vv = vv.transpose(1, 2, 0)


    # image = vv[0:300][0:300][:][:]
    # print(len(image[0]))

    image = vv#[0:300]#, 0:300]

    vv_slic = seg.slic(image)
    # plt.imshow(vv_slic)
    # plt.imsave('result_rgb.png', vv_slic)

    image_with_bounds = seg2.mark_boundaries(image=image, label_img=vv_slic)
    plt.imsave('bounded_result_rgb.png', image_with_bounds)