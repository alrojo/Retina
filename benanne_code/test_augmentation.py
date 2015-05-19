import numpy as np
import data

import matplotlib.pyplot as plt
plt.ion()

print "Load images"
images = data.load_images()
images_train = images[data.split['train']]
labels_train = data.labels_train[data.split['train']]

chunk_size = 256
num_chunks = 1

augmentation_params = {
    'rotation_range': (0, 360),
    'zoom_range': (1/1.1, 1.1),
}

print "Generate augmented images"
gen = data.gen_images_augmented(images_train, labels_train, num_chunks, chunk_size=chunk_size,
                                augmentation_params=augmentation_params, rng=np.random.RandomState(0))

gen_old = data.gen_images(images_train, labels_train, num_chunks, chunk_size=chunk_size, rng=np.random.RandomState(0))

chunk_x, chunk_y = gen.next()
chunk_x_old, chunk_y_old = gen_old.next()

print "Plot"
for k in xrange(chunk_size):
    plt.figure(1)
    plt.clf()
    plt.title("augmented")
    plt.imshow(chunk_x[k].transpose(1, 2, 0))

    plt.figure(2)
    plt.clf()
    plt.title("not augmented")
    plt.imshow(chunk_x_old[k].transpose(1, 2, 0))

    raw_input()
