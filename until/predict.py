import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import sys
import importlib
import os

import data

if not (2 <= len(sys.argv) <= 4):
    sys.exit("Usage: python predict.py <metadata_path> [subset=test] [config_name]")

metadata_path = sys.argv[1]

if len(sys.argv) >= 3:
    subset = sys.argv[2]
    assert subset in ['train', 'valid', 'test']
else:
    subset = 'test'

assert subset == 'test', "prediction generation for other subsets than test is not implemented yet."

print "Loading metadata file %s" % metadata_path

metadata = np.load(metadata_path)

if len(sys.argv) >= 4:
    config_name = sys.argv[3]
else:
    config_name = metadata['config_name']


config = importlib.import_module("configurations.%s" % config_name)

print "Using configuration: '%s'" % config_name

print "Build model"

l_in, l_out = config.build_model()

index = T.lscalar('index')
x = nn.utils.shared_empty(4)

all_params = nn.layers.get_all_params(l_out)

givens = {
    l_in.input_var: x[index * config.batch_size:(index + 1) * config.batch_size],
}

ratings = T.clip(l_out.get_output(deterministic=True), 0.0, 4.0)

print "Load parameters"

nn.layers.set_all_param_values(l_out, metadata['param_values'])

print "Compile functions"

predict = theano.function([index], ratings, givens=givens)


print "Predict"

gen = data.gen_chunks_from_gen_fixed(data.gen_test_images(), chunk_size=config.chunk_size)

predictions = []
num_test_examples = 0
for i, (chunk_x, chunk_length) in enumerate(gen):
    print "Chunk %d" % (i + 1)

    print "  transfering data to GPU"
    x.set_value(chunk_x)
    num_batches = int(np.ceil(chunk_length / float(config.batch_size)))

    print "  generate predictions"
    losses = []
    for b in xrange(num_batches):
        p = predict(b)
        predictions.append(p)

    num_test_examples += chunk_length

predictions = np.concatenate(predictions)
predictions = np.round(predictions[:num_test_examples])

predictions_path = os.path.join("predictions", os.path.basename(metadata_path).replace("dump_", "predictions_").replace(".pkl", ".npy"))

print "Storing predictions in %s" % predictions_path
np.save(predictions_path, predictions)
