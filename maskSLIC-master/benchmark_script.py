from datetime import datetime
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

import maskslic as seg
import skimage2.segmentation as seg2

#vv_src = rasterio.open('data/VV_VH.tif')
#vv = vv_src.read()
vv = imread('data/chelsea.png', as_gray=True    )#/VV_VH.tif')
# vv = vv[:3, :, :]
# vv = vv.transpose(1, 2, 0)

# image = vv[0:300][0:300][:][:]
# print(len(image[0]))



def bounded_test(vv):

    x1 = 300
    y1 = 200
    x2 = 500
    y2 = 400
    image = vv[x1:x2, y1:y2]

    # mean = np.mean(image)
    # image *= 1/mean

    k=1000

    vv_slic = seg.slic(image, n_segments=k, enforce_connectivity=False, max_iter=10)

    info = f'LADO_{str(x2-x1)}_K_{k}_{str(datetime.now())}'
    # plt.imshow(vv_slic)
    plt.imsave(f'export/result_{info}.png', vv_slic)

    image_with_bounds = seg2.mark_boundaries(image=image/np.max(image), label_img=vv_slic)
    plt.imsave(f'export/bounded_result_{info}.png', image_with_bounds)

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

bounded_test(vv)