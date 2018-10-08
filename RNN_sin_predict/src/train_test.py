import argparse
import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import matplotlib.pyplot as plt
import numpy as np
from chainer import (Chain, ChainList, Function, Link, Variable, datasets,
                     gradient_check, iterators, optimizers, report,
                     serializers, training, utils)
from chainer.backends import cuda
from chainer.training import extensions

from RNN import RNN


def load_dataset():
    x_data, t_data = [], []
    train_data = np.array([np.sin(i*2*np.pi/80) for i in range(80)])
    for i in range(len(train_data) - 1):
        x_data.append(np.array([train_data[i]], dtype=np.float32))
        t_data.append(np.array([train_data[i+1]], dtype=np.float32))

    return x_data, t_data


def main():
    n_in_out = 1 # input_out number
    n_units = 5 # hidden units number
    epochs = 100 # epoch number

    model = RNN(n_in_out, n_units)
    model.reset_state()
    optimizer = optimizers.SGD(lr=0.005)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(10.0))

    # load dataset
    x_data, t_data = load_dataset()

    # plot data
    plot_x, plot_y = [], []

    # remove result.txt
    if(os.path.exists('./result.txt')):
        os.remove('./result.txt')

    # -----Training-----
    for epoch in range(epochs):
        loss = 0
        
        for x, t in zip(x_data, t_data):
            x = np.array([x], dtype=np.float32)
            t = np.array([t], dtype=np.float32)
            loss += model(x=x, t=t, train=True)

        model.reset_state()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

        # plot data
        plot_x.append(epoch)
        plot_y.append(loss.data)
        
        # Test and save result
        print("epoch : %3d, loss %f" % (epoch, loss.data))
    # -----end Training-----

    # plot sin curve
    plt.title('Sine curve')
    plt.xlabel('t step')
    plt.ylabel('sin(t)')

    answerX = [i for i in range(80)]
    answerY = [np.sin(i*2*np.pi/80) for i in range(80)]
    plt.plot(answerX, answerY, color="blue", label="t")

    predictY = []
    predicted = model(np.array([[answerX[0]]], dtype=np.float32))
    predictY.append(answerY[0])
    for i in range(1, 80):
        if(0 <= i and i <= 5):
            predicted = model(np.array([[answerY[i]]], dtype=np.float32))
            predictY.append(answerY[i])
        else:
            predicted = model(np.array([[predictY[-1]]], dtype=np.float32))
            predictY.append(predicted.data[0])
    plt.plot(answerX, predictY, color="red", label="y")

    plt.plot(answerX[:5+1], answerY[:5+1], color="green", label="st10")

    plt.show()


if __name__ == '__main__':
    main()
