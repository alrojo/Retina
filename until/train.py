# THEANO_FLAGS=device=gpu3,mode=FAST_RUN,floatX=float32,optimizer_including=cudnn

import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import time
import sys
import importlib
import utils
from datetime import datetime, timedelta
import cPickle as pickle

import data
from buffering import buffered_gen_threaded
from quadratic_weighted_kappa import quadratic_weighted_kappa


if len(sys.argv) != 2:
    sys.exit("Usage: python train.py <config_name>")

config_name = sys.argv[1]
#config_name = 'alex'

config = importlib.import_module("configurations.%s" % config_name)

print "Using configuration: '%s'" % config_name

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (config_name, timestamp)
metadata_path = "/home/ubuntu/small_data_extra/metadata_%s" % experiment_id

print "Experiment id: %s" % experiment_id

print "Build model"

l_in, l_out = config.build_model()

# If loss function has not been specified set loss function to be root mean squared error
if hasattr(config, 'build_objective'):
    obj = config.build_objective(l_in, l_out)            # Create variable that calculates the loss function
else:
    obj = nn.objectives.Objective(l_out, loss_function=nn.objectives.mse)

index = T.lscalar('index')             # Initialse index
x = nn.utils.shared_empty(4)           # Create empty Theano shared variable with dimension 4
y = nn.utils.shared_empty(2)           # Create empty Theano shared variable with dimension 2

all_params = nn.layers.get_all_params(l_out)     # Get all parameters

if hasattr(config, 'set_weights'):
    nn.layers.set_all_param_values(l_out, config.set_weights())

# Importing learning rate schedule
if hasattr(config, 'learning_rate_schedule'):
    learning_rate_schedule = config.learning_rate_schedule              # Import learning rate schedule
else:
    learning_rate_schedule = { 0: config.learning_rate }

# Creating Theano shared variable to hold the learning rate
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

# Creating variable to evaluate training loss (note that deterministic=False due to the use of dropout)
train_loss = obj.get_loss(deterministic=False)

# Use nesterov momentum as update rule when using gradient descent
updates = nn.updates.nesterov_momentum(train_loss, all_params, learning_rate, config.momentum)

# Set givens parameter. Note that the givens parameter is set such that the whole training chunk can be loaded to the GPU
# and the only thing which is transfered between the CPU and GPU memory is the index denoting which batch of the data chunk
# the gradient should be calculated on. Hence, the givens parameter is used to denote which part of the data the update rule
# should be performed on.
givens = {
    l_in.input_var: x[index * config.batch_size:(index + 1) * config.batch_size],
    obj.target_var: y[index * config.batch_size:(index + 1) * config.batch_size],
}

# All ratings are clipped such that values less than zero are set equal to zero and values greater than 4 are
# set equal to 4
ratings = T.clip(l_out.get_output(deterministic=True), 0.0, 4.0)

# Compile training and evaluation functions (deterministic=True in eval function to disable random dropout).
iter_train = theano.function([index], obj.get_loss(deterministic=False), givens=givens, updates=updates)
iter_eval = theano.function([index], [obj.get_loss(deterministic=True), ratings], givens=givens)

print "Train"

if hasattr(config, 'create_train_gen'):
    create_train_gen = config.create_train_gen        # Assign data generating function
else:
    def create_train_gen():
        image_gen = data.gen_images(data.paths_my_train, data.labels_my_train, shuffle=True, repeat=True)
        chunks_gen = data.gen_chunks(image_gen, image_size=config.image_size, chunk_size=config.chunk_size, labels=True)
        # return chunks_gen
        return buffered_gen_threaded(chunks_gen)

if hasattr(config, 'create_eval_gen_train'):
    create_eval_gen_train = config.create_eval_gen_train
else:
    def create_eval_gen_train():
        image_gen = data.gen_images(data.paths_my_train, data.labels_my_train, shuffle=False, repeat=False)
        chunks_gen = data.gen_chunks(image_gen, image_size=config.image_size, chunk_size=config.chunk_size, labels=True)
        # return chunks_gen
        return buffered_gen_threaded(chunks_gen)

if hasattr(config, 'create_eval_gen_valid'):
    create_eval_gen_valid = config.create_eval_gen_valid
else:
    def create_eval_gen_valid():
        image_gen = data.gen_images(data.paths_my_valid, data.labels_my_valid, shuffle=False, repeat=False)
        chunks_gen = data.gen_chunks(image_gen, image_size=config.image_size, chunk_size=config.chunk_size, labels=True)
        # return chunks_gen
        return buffered_gen_threaded(chunks_gen)

# Number of batches in each chunk of data
num_batches_train = config.chunk_size // config.batch_size

# Start timers
start_time = time.time()
prev_time = start_time

all_losses_train = []
all_losses_eval_train = []
all_losses_eval_valid = []
all_kappas_train = []
all_kappas_valid = []
# Start training model
for i, (chunk_x, chunk_y, _) in enumerate(create_train_gen()):
    if i >= config.num_chunks:
        break

    print "Chunk %d of %d" % (i + 1, config.num_chunks)

    if i in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[i])
        print "  setting learning rate to %.7f" % lr
        learning_rate.set_value(lr)

    print "  transfering data to GPU"
    x.set_value(chunk_x)
    y.set_value(chunk_y)

    print "  train"
    losses = []
    for b in xrange(num_batches_train):
        loss = iter_train(b)
        losses.append(loss)

    loss_train = np.mean(losses)
    all_losses_train.append(loss_train)

    print "  average training loss: %.5f" % loss_train

    if (i + 1) % config.validate_every == 0:
        sets = [('train', create_eval_gen_train, data.labels_my_train, all_losses_eval_train, all_kappas_train),
                ('valid', create_eval_gen_valid, data.labels_my_valid, all_losses_eval_valid, all_kappas_valid)]

        for subset, gen, labels, all_losses, all_kappas in sets:
            print "  validating: %s loss" % subset
            losses = []
            ratings = []
            for j, (chunk_x, chunk_y, chunk_length) in enumerate(gen()):
                num_batches = chunk_length // config.batch_size # chop off the last (incomplete) batch
                x.set_value(chunk_x)
                y.set_value(chunk_y)

                for b in xrange(num_batches):
                    loss, r = iter_eval(b)
                    losses.append(loss)
                    ratings.append(r)

            loss_eval = np.mean(losses)
            all_losses.append(loss_eval)

            print "  average evaluation loss (%s): %.5f" % (subset, loss_eval)

            ratings = np.concatenate(ratings)
            ratings = np.round(ratings) # round to the nearest integer

            kappa = quadratic_weighted_kappa(ratings, labels[:len(ratings)], min_rating=0, max_rating=4)
            all_kappas.append(kappa)

            print "  kappa (%s): %.5f" % (subset, kappa)

    now = time.time()
    time_since_start = now - start_time
    time_since_prev = now - prev_time
    prev_time = now
    est_time_left = time_since_start * config.num_chunks
    eta = datetime.now() + timedelta(seconds=est_time_left)
    eta_str = eta.strftime("%c")
    print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
    print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
    print

    if (i + 1) % config.save_every == 0:
        print "  saving parameters and metadata"
        with open((metadata_path + "%d" % (round(i/config.save_every)) + ".pkl"), 'w') as f:
            pickle.dump({
                    'config_name': config_name,
                    'param_values': nn.layers.get_all_param_values(l_out),
                    'losses_train': all_losses_train,
                    'losses_eval_train': all_losses_eval_train,
                    'losses_eval_valid': all_losses_eval_valid,
                    'kappas_valid': all_kappas_valid,
                    'kappas_train': all_kappas_train,
                    'time_since_start': time_since_start,
                    'i': i,
                }, f, pickle.HIGHEST_PROTOCOL)

        print "  stored in %s" % metadata_path

    print