import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
import h5py

class deeperUNet(object):

    def __init__(self, input_dim=(1, 24, 24),num_classes=2, weight_scale=1e-2, reg=0.0005,
                 dtype=np.float32, convLayer=16, h5_file=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        C, H, W = input_dim
        #assert H == W
        num_filters = [1, 8, 8, 10, 10, 12, 12, 13,13,12, 12, 10, 10, 8, 8, 8, 2]
        #num_filters = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
        #num_filters = [1, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 4, 2]
        #num_filters = [1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 2]
        #num_filters = [1, 10, 20, 40, 40, 20, 10, 2]
        #num_filters = [1, 16, 32, 64, 64, 32, 16, 2]
        #num_filters = [1, 32, 64, 128, 128, 64, 32, 2]
        #num_filters = [1,2,3,4,4,3,2,2]
        kernal_size = [3,3,3,3,3,3,12,1,12,3,3,3,3,3,3,1]
        #kernal_size = [3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1]
        #kernal_size = [3, 3, 3, 3, 6, 1, 6, 3, 3, 3, 3, 1]
        #kernal_size = [3, 3, 3, 3, 3, 3, 1]
        num_filters[0] = C
        self.dim = input_dim
        self.optim_config = {}
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.kz = kernal_size
        self.convL = convLayer
        self.convLayer = convLayer
        self.numF = num_filters
        self.c = {}
        self.input_dim = input_dim
        self.h5_file = h5_file

        self.bnp = []
        self.bnp = [{'mode': 'train'} for i in xrange(self.convLayer)]
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # Intialization from Dafne:


        for i in range(convLayer):
            self.params['W%d' % i] = np.random.normal(0, weight_scale,(self.numF[i + 1], self.numF[i], self.kz[i], self.kz[i]))
            # F, c, HH, WW = w.shape
            self.params['b%d' % i] = np.zeros(self.numF[i + 1])
            if i < convLayer - 1:
                self.params['ga%d' % i] = np.ones(self.numF[i + 1])
                self.params['be%d' % i] = np.zeros(self.numF[i + 1])
                self.bnp[i]['running_mean'] = np.zeros(self.numF[i + 1])
                self.bnp[i]['running_var'] = np.zeros(self.numF[i + 1])
        self.params['W13'] = np.random.normal(0, weight_scale,(self.numF[14], self.numF[13] * 2, self.kz[13], self.kz[13]))
        self.params['W11'] = np.random.normal(0, weight_scale,(self.numF[12], self.numF[11] * 2, self.kz[11], self.kz[11]))
        self.params['W9'] = np.random.normal(0, weight_scale, (self.numF[10], self.numF[9] * 2, self.kz[9], self.kz[9]))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

        if h5_file is not None:
            self.load_weights(h5_file)

    def save_weights(self,h5_file,verbose=False):
        with h5py.File(h5_file, 'w') as hf:
            for i in range(self.convL):
                hf.create_dataset('W%d' % i, data=self.params['W%d' % i])
                hf.create_dataset('b%d' % i, data=self.params['b%d' % i])
                if i < self.convL - 1:
                    hf.create_dataset('ga%d' % i, data=self.params['ga%d' % i])
                    hf.create_dataset('be%d' % i, data=self.params['be%d' % i])
                    hf.create_dataset('running_mean%d' % i, data=self.bnp[i]['running_mean'])
                    hf.create_dataset('running_var%d' % i, data=self.bnp[i]['running_var'])

    '''
    for p, w in self.params.iteritems():
    config = self.optim_config[p]
    hf.create_dataset('_' + p, data=config['learning_rate'])
    hf.create_dataset('_' + p, data=config['be1'])
    hf.create_dataset('_' + p, data=config['be2'])
    hf.create_dataset('_' + p, data=config['epsilon'])
    hf.create_dataset('_' + p, data=config['m'])
    hf.create_dataset('_' + p, data=config['v'])
    hf.create_dataset('_' + p, data=config['t'])
    '''




    def load_weights(self, h5_file, verbose=False):
        import os.path
        print os.path.isfile(h5_file)
        if os.path.isfile(h5_file) == False:return
        with h5py.File(h5_file, 'r') as f:
            for k, v in f.iteritems():
                v = np.asarray(v)
                if k in self.params:
                    if verbose: print k, v.shape, self.params[k].shape
                    if v.shape == self.params[k].shape:
                        self.params[k] = v.copy()
                    elif v.T.shape == self.params[k].shape:
                        self.params[k] = v.T.copy()
                        raise ValueError('shapes for %s do not match exactly' % k)
                    else:
                        raise ValueError('shapes for %s do not match' % k)
                elif k.startswith('running_mean'):
                    i = int(k[12:])
                    assert self.bnp[i]['running_mean'].shape == v.shape
                    self.bnp[i]['running_mean'] = v.copy()
                    if verbose: print k, v.shape
                elif k.startswith('running_var'):
                    i = int(k[11:])
                    assert v.shape == self.bnp[i]['running_var'].shape
                    self.bnp[i]['running_var'] = v.copy()
                    if verbose: print k, v.shape
                else: print ' wrong '
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(self.dtype)

    def loss(self, X, y=None):
        N = X.shape[0]
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        #for bn_param in self.bnp:
        #    bn_param[mode] = mode
        #bn_param = self.bnp
        p = self.params
        convp1 = {'stride': 1, 'pad': 1}
        convp0 = {'stride': 1, 'pad': 0}
        convp5 = {'stride': 1, 'pad': self.kz[self.convL / 2] - 1}
        #print convp5
        scores = None
        #num_filters = [1, 8, 8, 16, 16, 32, 32, 16, 16, 8, 8, 2]
        poolp = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        shortCutX = {}
        for i in range(3):
            X, self.c['con%d'%(i*2)] = conv_bn_relu_forward(X, p['W%d'%(i*2)], p['b%d'%(i*2)], p['ga%d'%(i*2)], p['be%d'%(i*2)],convp1,self.bnp[i*2])
            X, self.c['con%d'%(i*2+1)] = conv_bn_relu_forward(X, p['W%d'%(i*2+1)], p['b%d'%(i*2+1)], p['ga%d'%(i*2+1)], p['be%d'%(i*2+1)],convp1,self.bnp[i*2+1])
            shortCutX['%d'%i] = X
            X, self.c['p%d',i] = max_pool_forward_fast(X, poolp)


        X, self.c['con6'] = conv_bn_relu_forward(X, p['W6'], p['b6'],p['ga6'], p['be6'],convp0,self.bnp[6])
        X, self.c['con7'] = conv_bn_relu_forward(X, p['W7'], p['b7'],p['ga7'], p['be7'],convp0,self.bnp[7])
        X, self.c['con8'] = deconv_bn_relu_forward(X, p['W8'], p['b8'], p['ga8'], p['be8'],convp5, self.bnp[8])

        for i in range(3):
            X = max_pool_backward_fast(X, self.c['p%d',2 - i])
            X, self.c['con%d'%(9+i*2)] = shortcut_deconv_bn_relu_forward(X,shortCutX['%d'%(2 - i)],p['W%d'%(9+i*2)], p['b%d'%(9+i*2)], p['ga%d'%(9+i*2)], p['be%d'%(9+i*2)],convp1, self.bnp[9+i*2])
            X, self.c['con%d'%(10+i*2)] = deconv_bn_relu_forward(X, p['W%d'%(10+i*2)], p['b%d'%(10+i*2)], p['ga%d'%(10+i*2)], p['be%d'%(10+i*2)],convp1, self.bnp[10+i*2])

        X, self.c['con15'] = conv_forward_fast(X, p['W15'], p['b15'], convp0)
        #N, C, H, W = x.shape
        #F, c, HH, WW = w.shape
        scores = X
        if y is None:
            return scores
        loss, d = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and d variables. Compute  #
        # data loss using softmax, and make sure that d[k] holds the gradients #
        # for p[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        reshaped_score = scores.transpose(0,2,3,1).reshape(-1,2)
        y = y.transpose(0,2,3,1).reshape(-1)
        '''
       Inputs:
          - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
            for the ith input.
          - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
            0 <= y[i] < C
       '''
        loss, dout = softmax_loss(reshaped_score, y,isize=96)
        dout[1] *= 30
        C,W,H = self.dim
        dout = dout.reshape(N,W,H,reshaped_score.shape[-1]).transpose(0,3,1,2)
        dout, d['W15'], d['b15'] = conv_backward_fast(dout, self.c['con15'])
        shortDout = {}
        for i in range(2,-1,-1):
            dout, d['W%d'%(10+i*2)], d['b%d'%(10+i*2)], d['ga%d'%(10+i*2)], d['be%d'%(10+i*2)] = deconv_bn_relu_backward(dout,self.c['con%d'%(10+i*2)])
            dout,shortDout['%d'%i], d['W%d'%(9+i*2)], d['b%d'%(9+i*2)], d['ga%d'%(9+i*2)], d['be%d'%(9+i*2)] = shortcut_deconv_bn_relu_backward(dout, self.c['con%d'%(9+i*2)])
            dout = max_upPooling_backward_reshape(dout, self.c['p%d',2 - i])


        dout, d['W8'], d['b8'], d['ga8'], d['be8'] = conv_bn_relu_backward(dout, self.c['con8'])
        dout, d['W7'], d['b7'], d['ga7'], d['be7'] = conv_bn_relu_backward(dout,self.c['con7'])
        dout, d['W6'], d['b6'], d['ga6'], d['be6'] = conv_bn_relu_backward(dout,self.c['con6'])

        for i in range(2, -1, -1):
            dout = max_pool_backward_fast(dout,self.c['p%d',i])
            dout, d['W%d'%(i*2+1)], d['b%d'%(i*2+1)], d['ga%d'%(i*2+1)], d['be%d'%(i*2+1)] = conv_bn_relu_backward(dout + shortDout['%d'%(2 - i)],self.c['con%d'%(i*2+1)])
            dout, d['W%d'%(i*2)], d['b%d'%(i*2)], d['ga%d'%(i*2)], d['be%d'%(i*2)] = conv_bn_relu_backward(dout, self.c['con%d'%(i*2)])
        for i in range(self.convL):
            reg_W_loss, dW_reg = self.regularization_loss(p['W%d' % i])
            d['W%d' % i] += dW_reg
            loss += reg_W_loss
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, d

    def regularization_loss(self, W):
        loss = self.reg * 0.5 * np.sum(W * W)
        dx = self.reg * W
        return loss, dx
