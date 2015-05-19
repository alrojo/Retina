# import time
# import os
# import sys
import numpy as np 
import theano
import theano.tensor as T
import lasagne as lag
import data
import time
import cPickle
import os
import datetime

# import importlib

# Use this when you start to keep a log

def train(config, N_train, exp_id, paths):
    config_name = exp_id
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = "%s-%s" % (config_name, timestamp)
    metadata_path = os.path.join(paths.metadata_dir, "dump_%s.pkl"
                                 % experiment_id)
    l_in, l_out = config.model()
    if hasattr(config, 'build_objective'):
        goal = config.objective(l_out)
    else:
        print "Found not model_objective using default."
        goal = lag.objectives.Objective(l_out,
                                        loss_functio=lag.objectives.mse)

    index = T.iscalar('index')
    # Todo: Figure out why the extra dimensions
    x = lag.utils.shared_empty(4)
    y = lag.utils.shared_empty(2)

    all_params = lag.layers.get_all_params(l_out)
    updates = lag.updates.nesterov_momentum(
        goal.get_loss(deterministic=False),
        all_params, learning_rate=config.learning_rate)
    # TODO: Makue sure givens only need these two values
    givens = {
        l_in.input_var: x[index * config.batch_size:(index + 1)
                          * config.batch_size],
        goal.target_var: y[index * config.batch_size:(index + 1)
                           * config.batch_size]}

    ratings = T.clip(l_out.get_output(deterministic=True), 0.0, 4.0)

    print "Compile functions"
    iter_train = theano.function([index], goal.get_loss(deterministic=False),
                                 givens=givens, updates=updates)
    iter_eval = theano.function([index], [goal.get_loss(deterministic=True),
                                ratings], givens=givens)

    print "Train"

    # TODO:  If time allows, look into buffered_gen_threaded(chunks_gen)
    if hasattr(config, 'create_train_gen'):
        create_train_gen = config.create_train_gen
    else:
        def create_train_gen():
            image_gen = data.gen_images(data.paths_my_train, data.labels_my_train, shuffle=True, repeat=True)
            chunks_gen = data.gen_chunks(image_gen, chunk_size=config.chunk_size, labels=True)
            return chunks_gen

    if hasattr(config, 'create_eval_gen_train'):
        create_eval_gen_train = config.create_eval_gen_train
    else:
        def create_eval_gen_train():
            image_gen = data.gen_images(data.paths_my_train, data.labels_my_train, shuffle=False, repeat=False)
            chunks_gen = data.gen_chunks(image_gen, chunk_size=config.chunk_size, labels=True)
            return chunks_gen

    if hasattr(config, 'create_eval_gen_valid'):
        create_eval_gen_valid = config.create_eval_gen_valid
    else:
        def create_eval_gen_valid():
            image_gen = data.gen_images(data.paths_my_valid, data.labels_my_valid, shuffle=False, repeat=False)
            chunks_gen = data.gen_chunks(image_gen, chunk_size=config.chunk_size, labels=True)
            return chunks_gen

    num_batches_train = config.chunk_size // config.batch_size

    start_time = time.time()

    all_losses_train = []
    all_losses_eval_train = []
    all_losses_eval_valid = []
    all_kappas_train = []
    all_kappas_valid = []

    for i, (chunk_x, chunk_y, _) in enumerate(create_train_gen()):
        if i >= config.num_chunks:
            break

        print "Chunk %d of %d" % (i + 1, config.num_chunks)

        print "  transfering data to GPU"
        x.set_value(chunk_x)
        y.set_value(chunk_y)


        print "train"
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

                kappa = fun.quadratic_weighted_kappa(ratings, labels[:len(ratings)], min_rating=0, max_rating=4)
                all_kappas.append(kappa)

                print "  kappa (%s): %.5f" % (subset, kappa)

        time_since_start = time.time() - start_time
        print "  time since start: %.2fs" % time_since_start

        if (i + 1) % config.save_every == 0:
            print "  saving parameters and metadata"
            
            with open(metadata_path, 'w') as f:
                cPickle.dump({
                        'config_name': config_name,
                        'param_values': lag.layers.get_all_param_values(l_out),
                        'losses_train': all_losses_train,
                        'losses_eval_train': all_losses_eval_train,
                        'losses_eval_valid': all_losses_eval_valid,
                        'kappas_valid': all_kappas_valid,
                        'kappas_train': all_kappas_train,
                        'time_since_start': time_since_start,
                        'i': i,
                    }, f, cPickle.HIGHEST_PROTOCOL)

            print "  stored in %s" % metadata_path


# print "Train: using config %"
# config = importlib.import_module("configurations.%s" % sys.arv[1])

if __name__ == '__main__':
    import fun
    fun.usage()
    exit()