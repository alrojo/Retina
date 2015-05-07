import numpy as np
import theano
import theano.tensor as T

from quadratic_weighted_kappa import confusion_matrix, quadratic_weighted_kappa

import nn_eyes


a = np.array([1, 3, 0, 4, 3, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0])
a_float = np.array([1, 3.2, 0, 4.3, 3.2, 0, 0, 0, 1.3, 0.8, 0, 0.5, 0, 2.2, 0, 0])
b = np.array([1, 4, 0, 4, 2, 0, 1, 0, 0, 1, 2, 0, 3, 2, 1, 0])

cm1 = np.array(confusion_matrix(a, b, 0, 4))
print cm1

th_a = theano.shared(a)
th_b = theano.shared(b)

cm2 = nn_eyes.continuous_conf_mat(th_a, th_b).eval()

print cm2

assert np.all(cm1 == cm2)



kappa1 = quadratic_weighted_kappa(a, b)
print kappa1

kappa2 = nn_eyes.continuous_weighted_kappa(th_a, th_b).eval()
print kappa2

assert kappa1 == kappa2
