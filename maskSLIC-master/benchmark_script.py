import rasterio
import numpy as np
import matplotlib.pyplot as plt

import maskslic as seg

vv_src = rasterio.open('data/VV_VH.tif')
vv = vv_src.read()
vv = vv[:3, :, :]
vv = vv.transpose(1, 2, 0)

vv_slic = seg.slic(vv, n_segments=10)
plt.imshow(vv_slic)
plt.imsave('result.png', vv_slic)
