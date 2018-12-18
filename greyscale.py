from PIL import Image
import os
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

for i in range(104, 400, 1):
    print(i)
    for j in range(2009):
        if os.path.exists('source/MCUCXR_0'+str(i)+'_1_'+str(j)+'.png'):
            image = Image.open('source/MCUCXR_0'+str(i)+'_1_'+str(j)+'.png').convert('L')
            im = numpy.array(image)
            plt.imshow(im, cmap='Greys_r')
            plt.imsave('source/MCUCXR_0'+str(i)+'_1_'+str(j)+'.png', im, cmap='Greys_r')

