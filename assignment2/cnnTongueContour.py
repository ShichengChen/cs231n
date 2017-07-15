from scipy.misc import imread, imsave, imresize
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.data_utils import load_CIFAR10
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

channels = 1
Xsz = 32
def loadData():
    region = 'E:/OpenCv_lib/BlockSnake/extract/region1/'
    _region = 'E:/OpenCv_lib/BlockSnake/extract/_region1/'
    Ndata = 203939
    Ndata = 2030
    #Ndata = 104085
    Tdata = 10001
    Tdata = 100
    Xtr = []
    Ytr = []
    Xte = []
    Yte = []
    for i in np.arange(0,Ndata,2):
        img0 = imread(region + str(i) + '.jpg')
        #img0 = cv2.cvtColor(img0,cv2.COLOR_GRAY2RGB)
        img0 = cv2.resize(img0,(Xsz,Xsz))
        img1 = img0.reshape(channels, Xsz, Xsz).astype("float")
        if i < Ndata - Tdata:
            Xtr.append(img1)
            Ytr.append(0)
        else:
            Xte.append(img1)
            Yte.append(0)
        img0 = imread(_region + str(i) + '.jpg')
        #img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
        img0 = cv2.resize(img0, (Xsz, Xsz))
        img2 = img0.reshape(channels, Xsz, Xsz).astype("float")
        if i < Ndata - Tdata:
            Xtr.append(img2)
            Ytr.append(1)
        else:
            Xte.append(img2)
            Yte.append(1)

    return np.array(Xtr),np.array(Ytr),np.array(Xte),np.array(Yte)

Xtr,Ytr,Xte,Yte = loadData()
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
  'X_train': Xtr,
  'y_train': Ytr,
  'X_val': Xte,
  'y_val': Yte,
}


model = ThreeLayerCNNNet(input_dim=(channels,Xsz,Xsz),weight_scale=1e-2,reg=1e-2,dropout=0.5)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                lr_decay=0.99,
                optim_config={
                  'learning_rate': 1e-4,
                    #1e-4
                    'beta2': 0.99999
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

plt.show()



for filenum in range(2):
    cap = cv2.VideoCapture(str(filenum + 1) + '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('mechineLearning' + str(filenum + 1) + '.avi',fourcc, 60.0, (320,240))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False: break
        image = frame.copy()
        image = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        height, width, depth = frame.shape

        for i in range(50 - 10, 120 + 10,5):
        #for i in range(0, height - Xsz, 5):
            for j in range(100 - 10, 200 + 10,5):
            #for j in range(0, width - Xsz, 5):
                X = np.array(image[i:i+Xsz,j:j + Xsz].copy())
                X = cv2.resize(X,(Xsz,Xsz))
                #print X.shape
                X1 = X.reshape((1, channels, Xsz, Xsz)).astype("float")
                X1 -= mean_image
                if(np.argmax(model.loss(X1), axis=1)[0] == 0):
                    cv2.circle(image,(j + Xsz / 2, i + Xsz / 2),2,(255,0,0))
        cv2.imshow('image',image)
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
