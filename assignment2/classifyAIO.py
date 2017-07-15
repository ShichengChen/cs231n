import numpy as np
import matplotlib.pyplot as plt
#from cs231n.classifiers.cnn import *
from cs231n.classifiers.threeLayerCNNNet import ThreeLayerCNNNet
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver
from scipy.misc import imread, imresize
from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time
from cs231n.classifiers.threeLayerCNNNet import *
import cv2

cap0 = cv2.VideoCapture('E:\\PythonProject\\train_data\\a.avi')
cap1 = cv2.VideoCapture('E:\\PythonProject\\train_data\\i.avi')
cap2 = cv2.VideoCapture('E:\\PythonProject\\train_data\\o.avi')
X_train = []
y_train = []
X_val = []
y_val = []
y_test = []
X_test = []
for i in range(7000):
    ret, frame0 = cap0.read()
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    frame0 = cv2.resize(frame0,(32,32))
    frame1 = cv2.resize(frame1, (32, 32))
    frame2 = cv2.resize(frame2, (32, 32))
    frame0 = np.transpose(frame0,(2,1,0))
    frame1 = np.transpose(frame1, (2, 1, 0))
    frame2 = np.transpose(frame2, (2, 1, 0))
    if i < 5000:
        X_train.append(frame0)
        y_train.append(0)
        X_train.append(frame1)
        y_train.append(1)
        X_train.append(frame2)
        y_train.append(2)
    elif i > 5000 and i < 6000:
        X_val.append(frame0)
        y_val.append(0)
        X_val.append(frame1)
        y_val.append(1)
        X_val.append(frame2)
        y_val.append(2)
    else:
        X_test.append(frame0)
        y_test.append(0)
        X_test.append(frame1)
        y_test.append(1)
        X_test.append(frame2)
        y_test.append(2)
print "readall"
data = {
      'X_train': np.array(X_train), 'y_train': np.array(y_train),
      'X_val': np.array(X_val), 'y_val': np.array(y_val),
      'X_test': np.array(X_test), 'y_test': np.array(y_test),
    }
num_train = 1000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'][:num_train],
  'y_val': data['y_val'][:num_train],
}
model = ThreeLayerCNNNet(weight_scale=1e-2,reg=1e-2)

solver = Solver(model, data,
                num_epochs=25, batch_size=2,
                update_rule='adam',
                lr_decay=0.99,
                optim_config={
                  'learning_rate': 1e-4,
                },
                verbose=True, print_every=100)
solver.train()