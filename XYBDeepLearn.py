import torch as tr
import numpy as np
from torch import autograd
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QRect

device_id = 0


def get_batch_list(length: int, mini_batch: int):
    index_copy = [x for x in range(0, length)]
    res = []
    while len(index_copy) > 0:
        res_one = []
        for i in range(0, mini_batch):
            if len(index_copy) <= 0:
                break
            index = index_copy[int(np.random.uniform(0, len(index_copy)))]
            res_one.append(index)
            index_copy.remove(index)
        res.append(res_one)
    return res


def activation_return(activation_type: str, data):
    res = 0
    if activation_type == 'sigmoid':
        res = tr.sigmoid(data)
    if activation_type == 'tan_h':
        res = tr.tanh(data)
    if activation_type == 'relu':
        res = tr.relu(data)
    if activation_type == 'softmax':
        res = tr.exp(data) / (tr.sum(tr.exp(data)) + 0.00001)
    return res


class XYBNeuron:
    w = None
    b = None
    x = None

    z = None
    a = None
    activation = 'sigmoid'

    next_neuron = None
    before_neron = None

    def __init__(self, *, w_dim: tuple, b_dim: tuple, activation: str):
        self.grad_b = 0
        self.grad_w = 0
        self.w = tr.randn(w_dim)
        self.b = tr.randn(b_dim)
        self.w = self.w.to(device_id)
        self.b = self.b.to(device_id)
        self.w.requires_grad_()
        self.b.requires_grad_()
        self.x = None

        self.a = None
        self.z = None
        self.activation = activation

        self.next_neuron = None
        self.before_neron = None

        self.record_w = 0
        self.record_b = 0
        self.mountain_w = 0
        self.mountain_b = 0

    def forward(self):
        self.z = tr.matmul(self.w, self.x) + self.b
        self.a = activation_return(self.activation, self.z)
        if self.next_neuron is not None:
            self.next_neuron.x = self.a
            self.next_neuron.forward()

    mountain_w = 0
    mountain_b = 0

    def grad_update(self, learn: float, mountain: float):
        self.mountain_w = self.mountain_w * mountain + (1 - mountain) * self.w.grad.data
        self.mountain_b = self.mountain_b * mountain + (1 - mountain) * self.b.grad.data

        self.w.data = self.w.data - self.mountain_w * learn
        self.b.data = self.b.data - self.mountain_b * learn

        self.record_w = 0
        self.record_b = 0
        self.w.grad.zero_()
        self.b.grad.zero_()


class XYBDeepLearning:
    neuron_list = []

    x_data = []
    y_data = []

    input_neuron = None
    output_neuron = None

    epoch = 0

    def __init__(self):
        tr.device('cuda', device_id)
        self.neuron_list = []
        self.x_data = []
        self.y_data = []
        self.input_neuron = None
        self.output_neuron = None

    def add_data(self, x, y):
        self.x_data.append(x.to(device_id))
        self.y_data.append(y.to(device_id))

    def add_neuron(self, *, neuron_num: int, neuron_type: str, activation: str):
        neuron = None
        b_dim = (neuron_num, 1)
        if neuron_type == 'input':
            w_dim = (neuron_num, self.x_data[0].shape[0])
            neuron = XYBNeuron(w_dim=w_dim, b_dim=b_dim, activation=activation)
            self.input_neuron = neuron
        if neuron_type == 'layer':
            w_dim = (neuron_num, self.neuron_list[-1].w.shape[0])
            neuron = XYBNeuron(w_dim=w_dim, b_dim=b_dim, activation=activation)
            self.neuron_list[-1].next_neuron = neuron
            neuron.before_neron = self.neuron_list[-1]
        if neuron_type == 'output':
            w_dim = (neuron_num, self.neuron_list[-1].w.shape[0])
            neuron = XYBNeuron(w_dim=w_dim, b_dim=b_dim, activation=activation)
            self.neuron_list[-1].next_neuron = neuron
            neuron.before_neron = self.neuron_list[-1]
            self.output_neuron = neuron

        self.neuron_list.append(neuron)

    def get_vector_by_minibatch(self, mini_batch):
        res_x = []
        res_y = []
        for batch in mini_batch:
            res_x.append(self.x_data[batch])
            res_y.append(self.y_data[batch])
        return tr.cat(res_x, dim=1), tr.cat(res_y, dim=1)

    def train(self, *, learn: float, mountain: float, mini_batch: int, decay_rate: float):
        self.epoch += 1
        coss = 0
        batch_list = get_batch_list(len(self.x_data), mini_batch)

        rate_learn = learn * (1 / (1 + (decay_rate * self.epoch)))

        for batch in batch_list:
            vector_data = self.get_vector_by_minibatch(batch)
            x = vector_data[0]
            y = vector_data[1]
            self.input_neuron.x = x
            self.input_neuron.forward()

            # Loss
            loss = y * (-1.0) * tr.log(self.output_neuron.a)
            coss += tr.sum(loss) / (loss.shape[0] * loss.shape[1])
            loss.backward(tr.ones(loss.shape, device=device_id), retain_graph=True)
            for neuron in self.neuron_list:
                neuron.grad_update(rate_learn, mountain)

        return coss / len(self.x_data)

    def get_result(self, x):
        self.input_neuron.x = x.to(device_id)
        self.input_neuron.forward()
        return self.output_neuron.z

