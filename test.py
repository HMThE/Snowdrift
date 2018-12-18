from PIL import Image
import os
import numpy
import matplotlib.pyplot as plt

image = Image.open('CXR_png/MCUCXR_0001_0.png')
im = numpy.array(image)

plt.imshow(im)
plt.imsave('f.png', im)
plt.show(image)

