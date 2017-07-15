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
from cs231n.classifiers.DeconvlutionNetwork import *
from cs231n.classifiers.deeperUNet import *
from cs231n.classifiers.UNet import *
from time import time
import cv2

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

'''
model = DeconNet()

N = 3
X = np.random.randn(N, 1, 24, 24)
y = np.random.randint(2, size=(N,1,24,24))

loss, grads = model.loss(X, y)
print 'Initial loss (no regularization): ', loss

model.reg = 0.5
loss, grads = model.loss(X, y)
print 'Initial loss (with regularization): ', loss
'''
'''
x = np.random.randn(2, 3, 6, 6)
w = np.random.randn(3, 3, 3, 3)
b = np.zeros(3,)
dout = np.random.randn(2, 3, 6, 6)
conv_param = {'stride': 1, 'pad': 1}

out, cache = deconv_forward(x, w, b, conv_param)
dx, dw, db = deconv_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: deconv_forward(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: deconv_forward(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: deconv_forward(x, w, b, conv_param)[0], b, dout)

print 'Testing conv_relu_pool'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)

'''

'''
from cs231n.layer_utils import *

def pd(x,w,b,gamma,beta,conv_p):
    p_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    x, pool_cache = max_pool_forward_fast(x,p_param)
    x,cache = deconv_bn_relu_unpooling_forward(x, w, b, gamma, beta, pool_cache, conv_p,{'mode':'train'})
    return x,cache,pool_cache
def pd_b(dout,cache,pcache):
    dout, dw, db, dgamma, dbeta = deconv_bn_relu_unpooling_backward(dout,cache)
    dx = max_pool_backward_fast(dout,pcache)
    return dx,dw,db,dgamma,dbeta


x = np.random.randn(2, 4, 4, 4)
#x = np.random.randn(2, 2, 2, 2)
w = np.random.randn(4, 4, 3, 3)
#w = np.random.randn(2, 2, 3, 3)
gamma = np.ones(4)
beta = np.zeros(4)
b = np.zeros(4,)
dout = np.random.randn(2, 4, 4, 4)
#dout = np.random.randn(2, 2, 2, 2)
conv_param = {'stride': 1, 'pad': 1}

out, cache,pcache = pd(x, w, b, gamma,beta,conv_param)
dx,dw,db,dgamma,dbeta = pd_b(dout, cache,pcache)

dx_num = eval_numerical_gradient_array(lambda x: pd(x, w, b,gamma,beta, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: pd(x, w, b,gamma,beta, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: pd(x, w, b,gamma,beta, conv_param)[0], b, dout)
dgamma_num = eval_numerical_gradient_array(lambda gamma: pd(x, w, b,gamma,beta, conv_param)[0], gamma, dout)
dbate_num = eval_numerical_gradient_array(lambda beta: pd(x, w, b,gamma,beta, conv_param)[0], beta, dout)

print 'Testing conv_relu_pool'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)
print 'dgamma error: ', rel_error(dgamma_num, dgamma)
print 'dbeta error: ', rel_error(dbate_num, dbeta)


'''
'''
num_inputs = 2
input_dim = (1, 48, 48)
reg = 1.25
num_classes = 2
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=(num_inputs,1,48,48))

model = deeperUNet(input_dim=(1, 48, 48),dtype=np.float64,reg=reg)
loss, grads = model.loss(X, y)
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))
'''


channels = 1
Xsz = 96

afterTest = 'G:/dataset/biggertest/'
afterTest = 'G:/dataset/originalTest2/'
afterTest = 'G:/dataset/cropedoriginalTest2/'
#afterTest = 'E:/PythonProject/dataset/bigtest/'
#Ndata = 5095
Ndata = 1903
Tdata = 1000

cap0 = cv2.VideoCapture('G:/movieDataset/gray1.avi')
cap1 = cv2.VideoCapture('G:/movieDataset/1label3.avi')

def loadmovie():
    Xtr = []
    Ytr = []
    Xte = []
    Yte = []
    i = 0
    while(True):
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if ret0 == False or ret1 == False:break
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        #print frame0.shape
        #print frame1.shape
        frame0 = frame0.reshape(channels, frame0.shape[0], frame0.shape[1])
        frame1 = frame1.reshape(channels, frame1.shape[0], frame1.shape[1]).astype("float")
        ret, thresh0 = cv2.threshold(frame0, 100, 1, cv2.THRESH_BINARY)
        #print thresh0.shape
        # ret, thresh1 = cv2.threshold(thresh0,0,255,cv2.THRESH_BINARY)
        # cv2.imshow("original", thresh0.reshape(Xsz,Xsz))
        # cv2.imshow("imgtest", thresh1.reshape(Xsz, Xsz))
        # cv2.waitKey(1)
        Y = thresh0.reshape(channels, frame0.shape[1], frame0.shape[2])
        i += 1
        if i < Tdata:
            Xte.append(frame1)
            Yte.append(Y)
        else:
            Xtr.append(frame1)
            Ytr.append(Y)
    return np.array(Xtr), np.array(Ytr), np.array(Xte), np.array(Yte)

Xtr,Ytr,Xte,Yte = loadmovie()
input_size = Xtr[0].shape[1]
print input_size
mean_image = np.mean(Xtr, axis=0)
print Xtr.shape
print Xte.shape
print Ytr.shape
print Yte.shape
Xtr -= mean_image
Xte -= mean_image

data = {
  'X_train': Xtr[:],
  'y_train': Ytr[:],
  'X_val': Xte[:],
  'y_val': Yte[:],
}

learning_rate = 0.01
reg = 0.0005

model = deeperUNet(input_dim=(Xtr[0].shape),h5_file = 'croped',reg = reg, dtype=np.float32)
#model = UNet(input_dim=(1, Xsz, Xsz),h5_file = None,reg = reg, dtype=np.float32)

solver = Solver(model, data,
                num_epochs=1, batch_size=8,
                update_rule='adam',
                lr_decay=1,
                max_jitter=0,
                h5_file = 'croped2',
                flipOrNot=True,
                optim_config={
                  'learning_rate': learning_rate,
                    #1e-4
                    'beta2': 0.999
                },
                verbose=True, print_every=1000)

solver.train()
plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')

#plt.show()



def writeData():
    cap0 = cv2.VideoCapture('G:/movieDataset/gray1.avi')
    cap1 = cv2.VideoCapture('G:/movieDataset/1label3.avi')
    while (True):
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        frame0 = frame0.reshape(1, 1, Xsz, Xsz)
        cv2.imshow("originalimgx",frame0)
        score = model.loss(frame0.astype("float") - mean_image)
        score = score.transpose(0, 2, 3, 1).reshape(-1, 2)
        ans = np.argmax(score, axis=1).reshape(Xsz, Xsz).astype("uint8")
        ret, ans = cv2.threshold(ans, 0, 255, cv2.THRESH_BINARY)
        cv2.imshow("ans", ans.reshape(Xsz, Xsz))
        cv2.waitKey(1)

writeData()

