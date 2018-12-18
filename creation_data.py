from PIL import Image
import os
import numpy
import matplotlib.pyplot as plt

for i in range(104, 400, 1):
    if os.path.exists('CXR_png/MCUCXR_0' + str(i) + '_1.png'):
        image = Image.open('CXR_png/MCUCXR_0' + str(i) + '_1.png').convert('L')
        im = numpy.array(image)
        tiles = []

        for x in range(0, im.shape[0], 100):
            for y in range(0, im.shape[1], 100):
                if x+100 > im.shape[0]:
                    tile = im[im.shape[0] - 100:im.shape[0], y:y+100]
                else:
                    if y + 100 > im.shape[1]:
                        tile = im[x:x+100, im.shape[1] - 100:im.shape[1]]
                    else:
                        tile = im[x:x+100, y:y+100]
                tiles.append(tile)
        del tiles[len(tiles)-1]
        tile = im[im.shape[0] - 100:im.shape[0], im.shape[1] - 100:im.shape[1]]
        tiles.append(tile)

        for j in range(len(tiles)):
            plt.imshow(tiles[j])
            plt.imsave('source/MCUCXR_0' + str(i) + '_1_' + str(j) + '.png', tiles[j])
