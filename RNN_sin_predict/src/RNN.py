import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import (Chain, ChainList, Function, Link, Variable, datasets,
                     gradient_check, iterators, optimizers, report,
                     serializers, training, utils)
from chainer.backends import cuda
from chainer.training import extensions


class RNN(chainer.Chain):
    def __init__(self, n_in_out, n_units):
        super(RNN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in_out, n_units)
            self.r1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, n_in_out)
        self.recurrent_h = None

    def reset_state(self):
        self.cleargrads()
        self.recurrent_h = None

    def __call__(self, x, t=None, train=False):
        if self.recurrent_h is None:
            self.recurrent_h = F.tanh(self.l1(x))
        else:
            self.recurrent_h = F.tanh(self.l1(x) + self.r1(self.recurrent_h))
        y = F.tanh(self.l2(self.recurrent_h))

        if(train):
            loss = F.mean_squared_error(y, t)
            return loss
        else:
            return y
