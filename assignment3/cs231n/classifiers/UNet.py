import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
import h5py

class UNet(object):

    def __init__(self, input_dim=(1, 24, 24),num_classes=2, weight_scale=1e-2, reg=0.0005,
                 dtype=np.float32, convLayer=12, h5_file=None):
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
        assert H == W
        num_filters = [1, 8, 8, 16, 16, 20, 20, 16, 16, 8, 8, 8, 2]
        #num_filters = [1, 4, 4, 8, 8, 16, 16, 8, 8, 4, 4, 4, 2]
        #num_filters = [1, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 4, 2]
        #num_filters = [1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 2]
        #num_filters = [1, 10, 20, 40, 40, 20, 10, 2]
        #num_filters = [1, 16, 32, 64, 64, 32, 16, 2]
        #num_filters = [1, 32, 64, 128, 128, 64, 32, 2]
        #num_filters = [1,2,3,4,4,3,2,2]
        kernal_size = [3,3,3,3,6,1,6,3,3,3,3,1]
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

        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in xrange(self.convLayer)]
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
                self.params['gamma%d' % i] = np.ones(self.numF[i + 1])
                self.params['beta%d' % i] = np.zeros(self.numF[i + 1])
                self.bn_params[i]['running_mean'] = np.zeros(self.numF[i + 1])
                self.bn_params[i]['running_var'] = np.zeros(self.numF[i + 1])
        self.params['W7'] = np.random.normal(0, weight_scale,(self.numF[8], self.numF[7] * 2, self.kz[7], self.kz[7]))
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
                    hf.create_dataset('gamma%d' % i, data=self.params['gamma%d' % i])
                    hf.create_dataset('beta%d' % i, data=self.params['beta%d' % i])
                    hf.create_dataset('running_mean%d' % i, data=self.bn_params[i]['running_mean'])
                    hf.create_dataset('running_var%d' % i, data=self.bn_params[i]['running_var'])

    '''
    for p, w in self.params.iteritems():
    config = self.optim_config[p]
    hf.create_dataset('_' + p, data=config['learning_rate'])
    hf.create_dataset('_' + p, data=config['beta1'])
    hf.create_dataset('_' + p, data=config['beta2'])
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
                    assert self.bn_params[i]['running_mean'].shape == v.shape
                    self.bn_params[i]['running_mean'] = v.copy()
                    if verbose: print k, v.shape
                elif k.startswith('running_var'):
                    i = int(k[11:])
                    assert v.shape == self.bn_params[i]['running_var'].shape
                    self.bn_params[i]['running_var'] = v.copy()
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
        #for bn_param in self.bn_params:
        #    bn_param[mode] = mode
        #bn_param = self.bn_params
        p = self.params
        convp1 = {'stride': 1, 'pad': 1}
        convp0 = {'stride': 1, 'pad': 0}
        convp5 = {'stride': 1, 'pad': self.kz[self.convL / 2] - 1}
        #print convp5
        scores = None
        #num_filters = [1, 8, 8, 16, 16, 32, 32, 16, 16, 8, 8, 2]
        poolp = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        X, self.c['con0'] = conv_bn_relu_forward(X, p['W0'], p['b0'], p['gamma0'], p['beta0'],convp1,self.bn_params[0])
        X, self.c['con1'] = conv_bn_relu_forward(X, p['W1'], p['b1'], p['gamma1'], p['beta1'],convp1,self.bn_params[1])
        shortCutX0 = X
        X, self.c['p0'] = max_pool_forward_fast(X, poolp)
        X, self.c['con2'] = conv_bn_relu_forward(X, p['W2'], p['b2'], p['gamma2'], p['beta2'], convp1,self.bn_params[2])
        X, self.c['con3'] = conv_bn_relu_forward(X, p['W3'], p['b3'],p['gamma3'], p['beta3'],convp1,self.bn_params[3])
        shortCutX1 = X
        X, self.c['p1'] = max_pool_forward_fast(X, poolp)
        X, self.c['con4'] = conv_bn_relu_forward(X, p['W4'], p['b4'],p['gamma4'], p['beta4'],convp0,self.bn_params[4])
        X, self.c['con5'] = conv_bn_relu_forward(X, p['W5'], p['b5'],p['gamma5'], p['beta5'],convp0,self.bn_params[5])
        X, self.c['con6'] = deconv_bn_relu_forward(X, p['W6'], p['b6'], p['gamma6'], p['beta6'],convp5, self.bn_params[6])
        X = max_pool_backward_fast(X, self.c['p1'])

        X, self.c['con7'] = shortcut_deconv_bn_relu_forward(X,shortCutX1,p['W7'], p['b7'], p['gamma7'], p['beta7'],convp1, self.bn_params[7])
        X, self.c['con8'] = deconv_bn_relu_forward(X, p['W8'], p['b8'], p['gamma8'], p['beta8'],convp1, self.bn_params[8])
        X = max_pool_backward_fast(X, self.c['p0'])

        X, self.c['con9'] = shortcut_deconv_bn_relu_forward(X,shortCutX0, p['W9'], p['b9'], p['gamma9'], p['beta9'],convp1, self.bn_params[9])
        X, self.c['con10'] = deconv_bn_relu_forward(X, p['W10'], p['b10'], p['gamma10'], p['beta10'], convp1, self.bn_params[10])
        X, self.c['con11'] = conv_forward_fast(X, p['W11'], p['b11'], convp0)
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
        loss, dout = softmax_loss(reshaped_score, y,isize=48)
        dout[1] *= 30
        C,W,H = self.dim
        dout = dout.reshape(N,W,H,reshaped_score.shape[-1]).transpose(0,3,1,2)
        dout, d['W11'], d['b11'] = conv_backward_fast(dout, self.c['con11'])
        dout, d['W10'], d['b10'], d['gamma10'], d['beta10'] = deconv_bn_relu_backward(dout,self.c['con10'])
        dout,shortDout1, d['W9'], d['b9'], d['gamma9'], d['beta9'] = shortcut_deconv_bn_relu_backward(dout, self.c['con9'])
        dout = max_upPooling_backward_reshape(dout, self.c['p0'])
        dout, d['W8'], d['b8'], d['gamma8'], d['beta8'] = deconv_bn_relu_backward(dout, self.c['con8'])
        dout,shortDout0, d['W7'], d['b7'], d['gamma7'], d['beta7'] =  shortcut_deconv_bn_relu_backward(dout, self.c['con7'])
        dout = max_upPooling_backward_reshape(dout, self.c['p1'])
        dout, d['W6'], d['b6'], d['gamma6'], d['beta6'] = conv_bn_relu_backward(dout, self.c['con6'])
        dout, d['W5'], d['b5'], d['gamma5'], d['beta5'] = conv_bn_relu_backward(dout,self.c['con5'])
        dout, d['W4'], d['b4'], d['gamma4'], d['beta4'] = conv_bn_relu_backward(dout,self.c['con4'])
        dout = max_pool_backward_fast(dout,self.c['p1'])
        dout, d['W3'], d['b3'], d['gamma3'], d['beta3'] = conv_bn_relu_backward(dout + shortDout0,self.c['con3'])
        dout, d['W2'], d['b2'], d['gamma2'], d['beta2'] = conv_bn_relu_backward(dout, self.c['con2'])
        dout = max_pool_backward_fast(dout, self.c['p0'])
        dout, d['W1'], d['b1'], d['gamma1'], d['beta1'] = conv_bn_relu_backward(dout + shortDout1,self.c['con1'])
        dout, d['W0'], d['b0'], d['gamma0'], d['beta0'] = conv_bn_relu_backward(dout, self.c['con0'])
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
