import numpy as np
import data

import matplotlib.pyplot as plt
plt.ion()

chunk_size = 256

augmentation_params = {
    'rotation_range': (0, 360),
    'zoom_range': (1/1.1, 1.1),
    'do_flip': True,
}

print "Generate augmented images"
gen = data.gen_train_images_augmented(chunk_size=chunk_size, augmentation_params=augmentation_params)

chunk_x, chunk_y, _ = gen.next()

print "Plot"
for k in xrange(chunk_size):
    plt.figure(1)
    plt.clf()
    plt.imshow(chunk_x[k].transpose(1, 2, 0))

    raw_input()
