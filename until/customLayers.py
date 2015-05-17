import lasagne as nn
import numpy as np
import theano
import theano.tensor as T
from lasagne.utils import floatX


def logloss(x, t):
    """
    Calculates the mean negative log likelihood for the output from OrderedLogitLayer.
    Note that the mean is used instead of the sum so that the learning rate is less
    dependent on the batch size.
    """
    t = T.cast(t, 'int64')
    # Due to numerical instability some of the values very close to zero sometimes become negative
    x = T.clip(x, 0.00001, 0.999999)

    return -T.mean( T.log(x)[T.arange(t.shape[0]), t] )


def mse(x, t):
    """
    Calculates the mean squared error of array of softmax probabilities
    """
    x = T.clip(x, 0.0, 4.0)
    return T.mean((x - t)**2)


class sortUniform(nn.init.Uniform):
    """
    Sorts the output from the uniform initializer
    """
    def sample(self, shape):
        return floatX( np.sort( np.random.uniform(low=0.1, high=3, size=shape) ) )


class OrderedLogitLayer(nn.layers.Layer):
    """
    Implements the ordered logistic regression to be used as an alternative
    output layer to the traditionally applied multinomial logistic regression
    output layer.
    """
    def __init__(self, input_layer, num_units, W=nn.init.Orthogonal(), theta=sortUniform()):
        super(OrderedLogitLayer, self).__init__(input_layer)

        self.num_units = num_units

        input_shape = self.input_layer.get_output_shape()
        self.num_inputs = int(np.prod(input_shape[1:]))

        self.W = self.create_param(W, (self.num_inputs, 1), name='W')
        self.theta = self.create_param(theta, (1, self.num_units - 1), name='theta')

    def get_params(self):
        return [self.W] + [self.theta]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        # Calculate dot product
        X_beta = T.dot(input, self.W)
        # Assign temporary theta variable in order to not mess with the shared variable properties of self.theta
        # when changing the broacasting pattern. If this isn't done the get_value() of self.theta is none existing
        # causing an error when saving params.
        theta_tmp = self.theta
        # Set broadcasting pattern for theta and X_beta
        theta_tmp = T.patternbroadcast(theta_tmp, (True, False))
        X_beta = T.patternbroadcast(X_beta, (False, True))
        # Subtract X_beta from theta
        diff = theta_tmp - X_beta

        # Use logistic cdf
        cdf = T.nnet.sigmoid(diff)

        # Calculate the correct probabilities
        prob1 = cdf[:, 0].dimshuffle(0, 'x')              # P[:, 0] = cdf[:, 0]
        prob2 = cdf[:, 1:] - cdf[:, 0:-1]                 # P[:, 1:-1] = cdf[:, 1:] - cdf[:, 0:-1]
        prob3 = T.ones((cdf.shape[0], 1)) - cdf[:, -1]    # P[:, -1] = 1 - cdf[:, -1]

        # Concatenate matrices
        prob = T.horizontal_stack(prob1, prob2, prob3)

        return prob
