import theano
import theano.tensor as T

def cutoff(x):
    return T.clip(x, 0, 4)

def scaled_sigmoid(x, margin=0.5):
    return T.nnet.sigmoid(x) * (4 + 2 * margin) - margin


num_ratings = 5

def continuous_one_hot(x):
    columns = []

    # first rating (0)
    c0 = T.switch(x < 0, 1, T.switch(x < 1, 1 - (x - 0), 0))
    columns.append(c0)

    # middle ratings (1, 2, 3)
    for r in xrange(1, num_ratings - 1):
        c = T.switch(x < r  - 1, 0, T.switch(x < r, 1 - (r - x), T.switch(x < r + 1, 1 - (x - r), 0)))
        columns.append(c)

    # last rating (4)
    c4 = T.switch(x < 4 - 1, 0, T.switch(x < 4, 1 - (4 - x), 1))
    columns.append(c4)

    return T.stack(columns).T


def discrete_one_hot(x):
    x = T.cast(x, 'int32')
    return T.eye(num_ratings)[x]


def continuous_conf_mat(rater_a, rater_b):
    """
    rater_a: continuous predictions
    rater_b: labels (integers)
    """
    # rater_a_oh = T.eye(num_ratings)[rater_a]
    rater_a_oh_fractional = continuous_one_hot(rater_a)
    rater_b_oh = discrete_one_hot(rater_b)
    conf_mat = T.dot(rater_a_oh_fractional.T, rater_b_oh)
    return conf_mat


def discrete_histogram(x):
    return T.sum(discrete_one_hot(x), axis=0)


def continuous_histogram(x):
    return T.sum(continuous_one_hot(x), axis=0)


def continuous_weighted_kappa(rater_a, rater_b):
    num_scored_items = rater_a.shape[0]
    conf_mat = continuous_conf_mat(rater_a, rater_b)

    hist_rater_a = continuous_histogram(rater_a)
    hist_rater_b = discrete_histogram(rater_b)

    expected_counts = T.dot(hist_rater_a[:, None], hist_rater_b[None, :]) / num_scored_items

    i = T.cast(T.arange(num_ratings), 'float32')
    j = T.cast(T.arange(num_ratings), 'float32')
    weight_matrix = (i[:, None] - j[None, :]) ** 2 / (num_ratings - 1) ** 2

    kappa =  1 - T.sum(weight_matrix * conf_mat) / T.sum(weight_matrix * expected_counts)
    return kappa