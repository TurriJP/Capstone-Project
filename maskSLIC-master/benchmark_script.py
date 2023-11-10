import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import rasterio as rio

import maskslic as seg
import skimage.segmentation as seg2

#vv_src = rasterio.open('data/VV_VH.tif')
#vv = vv_src.read()
vv = imread('data/chelsea.png')
# vv = vv[:3, :, :]
# vv = vv.transpose(1, 2, 0)

# image = vv[0:300][0:300][:][:]
# print(len(image[0]))
# image = vv[0:500, 0:500]#, 0:300]
# print('imagem gato:')
# print(image.shape)

from rasterio.plot import reshape_as_image
# Lendo resultado
with rio.open('/media/jp/FreeAgent GoFlex Drive/TCC/MAIN2.tif') as src:
  image = src.read()
image = reshape_as_image(image)
print('imagem radar:')
print(image.shape)
# image = np.where(image == 0, np.random.uniform(0.005, 0.017), image)
# Create a mask for zeros in the array
zero_mask = (image == 0)

# Generate random numbers with the same shape as the imageay
random_numbers = np.random.rand(*image.shape)  # Use np.random.randn for random numbers from a standard normal distribution

# Replace zeros with random numbers
image[zero_mask] = random_numbers[zero_mask]


vv_slic = seg.slic(image, n_segments=2000, max_iter=100, compactness=0.1, enforce_connectivity=True  )
# plt.imshow(vv_slic)
plt.imsave('result.png', vv_slic)

image_with_bounds = seg2.mark_boundaries(image=image[:, :, 3]/np.max(image[:, :, 3]), label_img=vv_slic)
plt.imsave('bounded_result.png', image_with_bounds)
