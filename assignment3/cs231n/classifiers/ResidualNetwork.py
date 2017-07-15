import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
import h5py

class ResNet(object):

    def __init__(self, input_dim=(3, 32, 32),num_classes=10, reg=0.0001,
                 dtype=np.float32, convLayer=3, h5_file=None):
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

        if convLayer <= 3: weight_scale = [np.sqrt(2.0 / (9 * 3)),np.sqrt(2.0 / (9 * 8)),
                        np.sqrt(2.0 / (9 * 16)),np.sqrt(2.0 / (9 * 32)),np.sqrt(2.0 / 64)]
        else: weight_scale = [np.sqrt(2.0 / (9 * 3)),np.sqrt(2.0 / (9 * 8)),np.sqrt(2.0 / (9 * 16)),
                        np.sqrt(2.0 / (9 * 16)),np.sqrt(2.0 / (9 * 32)),np.sqrt(2.0 / (9 * 32)),np.sqrt(2.0 / (9 * 64)),
                              np.sqrt(2.0 / 64)]

        weight_scale = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        C, H, W = input_dim
        assert H == W
        if convLayer <= 3: num_filters = [8, 16, 32, 64]
        else: num_filters = [8,16, 16,32,32,64, 64]

        if convLayer <= 3:featureMapSize = [H,H / 2,H / 4]
        else: featureMapSize = [H,H, H / 2,H / 2, H / 4,H / 4]
        self.optim_config = {}
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.convL = convLayer
        self.convLayer = convLayer
        self.fSize = featureMapSize
        self.numF = num_filters
        self.cache = {}
        self.input_dim = input_dim
        self.h5_file = h5_file

        self.bn_params = []
        self.bn_params = [{'mode': 'train'} for i in xrange(self.convLayer*2)]
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


        self.params['W'] = np.random.normal(0, weight_scale[0], (num_filters[0], 3, 3, 3))
        self.params['b'] = np.zeros(8)
        for i in range(convLayer):
            self.params['W%d'%(i*2)] = np.random.normal(0, weight_scale[i + 1], (self.numF[i+1], self.numF[i], 3, 3))
            self.params['b%d'%(i*2)] = np.zeros(self.numF[i+1])
            self.params['gamma%d'%(i*2)] = np.ones(self.numF[i + 1])
            self.params['beta%d'%(i*2)] = np.zeros(self.numF[i+1])
            self.bn_params[i*2]['running_mean'] = np.zeros(self.numF[i+1])
            self.bn_params[i*2]['running_var'] = np.zeros(self.numF[i + 1])
            self.params['W%d'%(i*2+1)] = np.random.normal(0, weight_scale[i + 1], (self.numF[i+1], self.numF[i+1], 3, 3))
            self.params['b%d'%(i*2+1)] = np.zeros(self.numF[i+1])
            self.params['gamma%d'%(i*2+1)] = np.ones(self.numF[i+1])
            self.params['beta%d'%(i*2+1)] = np.zeros(self.numF[i+1])
            self.bn_params[i*2+1]['running_mean'] = np.zeros(self.numF[i + 1])
            self.bn_params[i*2+1]['running_var'] = np.zeros(self.numF[i + 1])
        self.params['W%d' % (convLayer*2+1)] = \
            np.random.normal(0, weight_scale[convLayer + 1],(self.numF[convLayer],num_classes))
        self.params['b%d' % (convLayer*2+1)] = np.zeros(num_classes)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

        if h5_file is not None:
            self.load_weights(h5_file)

    def save_weights(self,h5_file,verbose=False):
        with h5py.File(h5_file, 'w') as hf:
            hf.create_dataset('W', data=self.params['W'])
            hf.create_dataset('b', data=self.params['b'])
            for i in range(self.convL):
                hf.create_dataset('W%d' % (i * 2), data=self.params['W%d' % (i * 2)])
                hf.create_dataset('b%d' % (i * 2), data=self.params['b%d' % (i * 2)])
                hf.create_dataset('W%d' % (i * 2+1), data=self.params['W%d' % (i * 2+1)])
                hf.create_dataset('b%d' % (i * 2+1), data=self.params['b%d' % (i * 2+1)])
                hf.create_dataset('gamma%d' % (i * 2), data=self.params['gamma%d' % (i * 2)])
                hf.create_dataset('gamma%d' % (i * 2+1), data=self.params['gamma%d' % (i * 2+1)])
                hf.create_dataset('beta%d' % (i * 2), data=self.params['beta%d' % (i * 2)])
                hf.create_dataset('beta%d' % (i * 2 + 1), data=self.params['beta%d' % (i * 2 + 1)])
                hf.create_dataset('running_mean%d' % (i * 2), data=self.bn_params[i * 2]['running_mean'])
                hf.create_dataset('running_var%d' % (i * 2), data=self.bn_params[i * 2]['running_var'])
                hf.create_dataset('running_mean%d' % (i * 2+1), data=self.bn_params[i * 2 + 1]['running_mean'])
                hf.create_dataset('running_var%d' % (i * 2+1), data=self.bn_params[i * 2 + 1]['running_var'])
            hf.create_dataset('W%d' % (self.convL * 2 + 1), data=self.params['W%d' % (self.convL * 2 + 1)])
            hf.create_dataset('b%d' % (self.convL * 2 + 1), data=self.params['b%d' % (self.convL * 2 + 1)])



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
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        #for bn_param in self.bn_params:
        #    bn_param[mode] = mode
        #bn_param = self.bn_params

        scores = None
        X, self.cache['con'] = conv_forward_fast(X, self.params['W'],self.params['b'],conv_param={'stride': 1, 'pad': 1})
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        for i in range(self.convL):
            W0, b0 = self.params['W%d' % (i*2)], self.params['b%d' % (i*2)]
            W1, b1 = self.params['W%d' % (i * 2 + 1)], self.params['b%d' % (i * 2 + 1)]
            gamma0, beta0 = self.params['gamma%d' % (i * 2)], self.params['beta%d' % (i * 2)]
            gamma1, beta1 = self.params['gamma%d' % (i * 2 + 1)], self.params['beta%d' % (i * 2 + 1)]
            conv_param = {'stride': 1, 'pad': 1}
            X,self.cache['con%d'%i] = conv_bn_relu_conv_bn_byway_relu_forward(X,(W0,b0,W1,b1),
                                                                              (gamma0,beta0,gamma1,beta1),conv_param,
                                                                              (self.bn_params[i*2],self.bn_params[i*2+1]))
            if i != self.convL - 1:
                if (self.convL > 3 and i % 2 != 0) or (self.convL <= 3):
                    X,self.cache['pool%d'%i] = max_pool_forward_fast(X,pool_param)
            elif i == self.convL - 1:
                assert X.shape[3] == self.fSize[self.convL - 1]
                assert X.shape[2] == self.fSize[self.convL - 1]
                X = X.sum(axis=(2,3)) / (self.fSize[self.convL - 1] * self.fSize[self.convL - 1])

        W, b = self.params['W%d' % (self.convL*2+1)], self.params['b%d' % (self.convL*2+1)]
        scores, self.cache['con' + str(self.convL+1)] = affine_forward(X, W, b)

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        softmax_l, dout = softmax_loss(scores, y)
        dout, dW_fc, db_fc = affine_backward(dout, self.cache['con' + str(self.convL+1)])
        reg_W_loss, dW_reg = self.regularization_loss(self.params['W%d' % (self.convL*2+1)])
        grads['W%d' % (self.convL*2+1)] = dW_fc + dW_reg
        grads['b%d' % (self.convL*2+1)] = db_fc
        loss += softmax_l + reg_W_loss
        for i in range(self.convL - 1,-1,-1):
            if i != self.convL - 1:
                if (self.convL > 3 and i % 2 != 0) or (self.convL <= 3):
                    dout = max_pool_backward_fast(dout,self.cache['pool%d'%i])
            elif i == self.convL - 1:
                newshapedout = np.reshape(dout,(dout.shape[0],dout.shape[1],1,1))
                newshapezeros = np.zeros((dout.shape[0],dout.shape[1],self.fSize[self.convL-1],self.fSize[self.convL-1]))
                ans = newshapezeros + newshapedout
                dout = ans / (self.fSize[self.convL-1] * self.fSize[self.convL-1])
            dout,dwb,dgb = conv_bn_relu_conv_bn_byway_relu_backward(dout,self.cache['con%d'%i])
            dw0,db0,dw1,db1 = dwb
            dgamma0, dbeta0, dgamma1, dbeta1 = dgb
            reg_W_loss0, dW_reg0 = self.regularization_loss(self.params['W%d' % (i*2)])
            reg_W_loss1, dW_reg1 = self.regularization_loss(self.params['W%d' % (i * 2 + 1)])
            grads['gamma%d' % (i * 2)] = dgamma0
            grads['gamma%d' % (i * 2+1)] = dgamma1
            grads['beta%d' % (i * 2)] = dbeta0
            grads['beta%d' % (i * 2+1)] = dbeta1
            grads['W%d' % (i*2)] = dW_reg0 + dw0
            grads['W%d' % (i * 2 + 1)] = dW_reg1 + dw1
            grads['b%d' % (i*2)] = db0
            grads['b%d' % (i*2+1)] = db1
            loss += (reg_W_loss0 + reg_W_loss1)

        dout, grads['W'], grads['b'] = conv_backward_fast(dout, self.cache['con'])
        reg_W_loss, dW_reg = self.regularization_loss(self.params['W'])
        grads['W'] += dW_reg
        loss += reg_W_loss
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def regularization_loss(self, W):
        loss = self.reg * 0.5 * np.sum(W * W)
        dx = self.reg * W
        return loss, dx
