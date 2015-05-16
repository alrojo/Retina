import buffering
import numpy as np
import data

num_chunks = 4000
chunk_size = 16384
batch_size = 32

image_size = (96, 96)

# Set augmentation parameters
augmentation_params = {
    'zoom_range': (1 / 1.1, 1.1),
    'rotation_range': (0, 360),
}

meanAndStd = np.load('data/meanAndStd.npy')

image_gen = data.gen_images(data.paths_my_train, meanAndStd, data.labels_my_train, shuffle=True, repeat=True)


def augmented_image_gen():
        for image, label in image_gen:
            yield data.augment_image(image, image_size, augmentation_params), label

chunks_gen = data.gen_chunks(augmented_image_gen(), image_size,chunk_size=chunk_size, labels=True)

for chunk in chunks_gen:
    output = buffering.buffered_gen_threaded(chunk)

