import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import rasterio as rio
from rasterio.plot import reshape_as_image
import maskslic as seg
import skimage.segmentation as seg2

slic_dir = '/home/jp/Pictures/new_slic'
file = '/home/jp/Downloads/AREA_3_mat.tif'

def rescale_img(src, w, h):
  data = src.read(
      out_shape=(
          src.count,
          w,
          h
      ),
      )

  transform = src.transform * src.transform.scale(
      (src.width / data.shape[-1]),
      (src.height / data.shape[-2])
  )

  return data

def color_img():
  color_image = '/home/jp/Documents/IME/TCC/NEW/areas_jpg/x03_fcu.png'
  grayscale_src = rasterio.open(color_image)
  grayscale = grayscale_src.read()
  grayscale = grayscale.transpose(1, 2, 0)
  return grayscale

def test(filepath, n_segments, enforce_connectivity, distance, spatial_weight, filename='file', max_iter=10, compactness=0.1, color_image=None, w=None, h=None):
  with rio.open(filepath) as src:
      if w and h:
          image = rescale_img(src, w, h)
      else:
          image = src.read()
      image = reshape_as_image(image)

      # Remover valores iguais a 0 para evitar problemas
      zero_mask = (image == 0)
      random_numbers = np.random.rand(*image.shape)
      image[zero_mask] = random_numbers[zero_mask]
      print('antes do slic')
      print(type(max_iter))
      vv_slic = seg.slic(image, n_segments=n_segments, max_iter=max_iter, compactness=compactness, enforce_connectivity=True, distance=distance, spatial_weight=spatial_weight)
      print('antes do mark')
      plt.imsave(filename+'.png', vv_slic)
      if color_image is None:
          display_image = image[:, :, 2]
      else:
          display_image = color_image
      image_with_bounds = seg2.mark_boundaries(image=display_image, label_img=vv_slic)
      print('depois do mark')
      plt.imshow(image_with_bounds)
      plt.imsave(filename+'.png', image_with_bounds)
      np.save(slic_dir+filename+'.npy', vv_slic)
      return vv_slic
    
def main(color_img):
  filename = 'AREA3_c_10000_RETESTANDO'
  slic = test(file, 10000, True, 0, 5, filename, color_image=color_img, w=998, h=977) 
   
img = color_img()
main(img)










































def old_pipeline():

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


  vv_slic = seg.slic(image, n_segments=2000, max_iter=10, compactness=0.1, enforce_connectivity=True, distance=3, spatial_weight=5)
  # plt.imshow(vv_slic)
  plt.imsave('result.png', vv_slic)

  image_with_bounds = seg2.mark_boundaries(image=image[:, :, 3]/np.max(image[:, :, 3]), label_img=vv_slic)
  plt.imsave('bounded_result.png', image_with_bounds)
