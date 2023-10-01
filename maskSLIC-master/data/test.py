import sys
import numpy as np
sys.path.append('/home/jp/.snap/snap-python')


from snappy import ProductIO
p = ProductIO.readProduct('subset2/subset_2_of_S1A_IW_GRDH_1SDV_20150509T085850_20150509T085915_005842_007849_F3E8.dim')

vh=p.getBand('Amplitude_VH')
print(vh)
databox2=np.zeros(100*100)
# line = [0] * 100
# databox2 = line * 100

box2=vh.readPixels(100,100,100,100,databox2) 

print(box2)