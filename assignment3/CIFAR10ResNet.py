import numpy as np
import matplotlib.pyplot as plt
#from cs231n.classifiers.cnn import *
import cs231n.classifiers.ResidualNetwork
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver
from cs231n.classifiers.ResidualNetwork import *
from scipy.misc import imread, imresize
from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

'''
model = ResNet()

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print 'Initial loss (no regularization): ', loss

model.reg = 0.5
loss, grads = model.loss(X, y)
print 'Initial loss (with regularization): ', loss

num_inputs = 2
input_dim = (3, 16, 16)
reg = 0.01
num_classes = 3
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = ResNet(input_dim=input_dim, dtype=np.float64,reg=reg,convLayer=6)
loss, grads = model.loss(X, y)
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))

'''

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

num_train = 10
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}


data = {
  'X_train': data['X_train'][:],
  'y_train': data['y_train'][:],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
  'y_test': data['y_test'],
  'X_test':data['X_test'],
}
print data['y_test'].shape
print data['X_test'].shape

for iter in range(1):
    #TODO
    regr = 1e-4
    #regr = 0
    learning_rate = 1e-4
    model = ResNet(reg=regr,dtype=np.float64,h5_file='momentumresnet3',convLayer=3)

    solver = Solver(model, data,
                    num_epochs=100, batch_size=64,
                    update_rule='sgd_momentum',
                    lr_decay=0.99,
                    max_jitter=2,
                    h5_file = 'momentumresnet3',
                    flipOrNot=True,
                    optim_config={
                      'learning_rate': learning_rate,
                    },
                    verbose=True, print_every=1000)

    solver.train()
    #actrain = solver.check_accuracy(data['X_train'],data['y_train'])
    acxval = solver.check_accuracy(data['X_val'], data['y_val'])
    actest = solver.check_accuracy(data['X_test'], data['y_test'])
    print learning_rate,regr
    print acxval,actest
    print
    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, 'o',label=str(learning_rate) + ' ' + str(regr))
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o',label='train:' + str(learning_rate) + ' ' + str(regr))
    plt.plot(solver.val_acc_history, '-o',label='val:' + str(learning_rate) + ' ' + str(regr))
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('logg_acc.png')
    plt.pause(1)

plt.show()


