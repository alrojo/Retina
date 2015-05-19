import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
import glob


#images_david = glob.glob("/home/morten/Dropbox/Advanced_Machine_Learning/Retina/data/experiment_data/*.png")
#images_sample = glob.glob("/home/morten/Dropbox/Advanced_Machine_Learning/Retina/data/sample/*.jpeg")                                  
#images_pil_512_trans = glob.glob("/home/morten/Dropbox/Advanced_Machine_Learning/Retina/data/pil_512_trans/*.png")                                  
#images_pil_512_trans_jpg = glob.glob("/home/morten/Dropbox/Advanced_Machine_Learning/Retina/data/pil_512_trans_jpg/*.jpeg")     


images_sample = "/home/morten/Dropbox/Advanced_Machine_Learning/Retina/data/sample/15_left.jpeg"
images_pil_512_trans = "/home/morten/Dropbox/Advanced_Machine_Learning/Retina/data/pil_512_trans/15_left.png"
images_pil_512_trans_jpg = "/home/morten/Dropbox/Advanced_Machine_Learning/Retina/data/pil_512_trans_jpg/15_left.jpeg"


image = io.imread(images_sample)
image_pil = io.imread(images_pil_512_trans)
image_pil_jpg = io.imread(images_pil_512_trans_jpg)




print image.nbytes/1e6
print image_pil.nbytes/1e6
print image_pil_jpg.nbytes/1e6


plt.figure("HD")
(io.imshow(image))

plt.figure("png")
io.imshow(image_pil)

plt.figure("pil_jpeg")
io.imshow(image_pil_jpg)

plt.show()
#plt.show(io.imshow(image),block=True)
#plt.figure(2)
#plt.show(io.imshow(image_pil))

#plt.figure(3)
#plt.show(io.imshow(image_pil_jpg))
