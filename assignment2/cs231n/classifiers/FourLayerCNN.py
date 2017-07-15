import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class FourLayerCNN(object):


    def __init__(self, input_dim=(3, 32, 32), num_filters=[3,16,20,20,20],
                   convW=[3,3,3,3],seed=None,dropout=0,
                 num_classes=10, weight_scale=1e-2, reg=0.0,
                 dtype=np.float32, conv_relu_pool_Layer=4, fc_netLayer=1):
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
        self.params = {}
        self.reg = reg
        self.use_dropout = dropout > 0
        self.dtype = dtype
        self.convL = conv_relu_pool_Layer
        self.Alayer = conv_relu_pool_Layer + 1
        self.filter_size = convW
        self.cache = {}
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
        C, H, W = input_dim


        num_filters[0] = C
        for i in range(conv_relu_pool_Layer):
            self.params['W%d' % i] = np.random.normal(0, weight_scale,
                                                 (num_filters[i + 1], num_filters[i], convW[i], convW[i]))
            self.params['b%d' % i] = np.zeros(num_filters[i + 1])
            H /= 2
            W /= 2
        self.params['W%d' % conv_relu_pool_Layer] = \
            np.random.normal(0, weight_scale,(H * W * num_filters[conv_relu_pool_Layer],num_classes))
        self.params['b%d' % conv_relu_pool_Layer] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        scores = None
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        for i in range(self.convL):
            W, b = self.params['W%d' % i], self.params['b%d' % i]
            conv_param = {'stride': 1, 'pad': (self.filter_size[i] - 1) / 2}
            X,self.cache['conv_out' + str(i)],self.cache['relu_out' + str(i)],self.cache['con' + str(i)]\
                = conv_relu_pool_forward(X, W, b, conv_param, pool_param)
            self.cache['pool_out' + str(i)] = X
        if self.use_dropout:
            X, self.cache['dropout'] = dropout_forward(X, self.dropout_param)
        W, b = self.params['W%d' % self.convL], self.params['b%d' % self.convL]
        scores, self.cache['con' + str(self.convL)] = affine_forward(X, W, b)

        if y is None:
            return scores

        loss, grads = 0, {}
        for i in range(self.convL + 1):
            grads['W%d' % i] = 0
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        softmax_l, dout = softmax_loss(scores, y)
        dout, dW_fc, db_fc = affine_backward(dout, self.cache['con' + str(self.convL)])
        reg_W_loss, dW_reg = self.regularization_loss(self.params['W%d' % self.convL])
        grads['W%d' % self.convL] = dW_fc + dW_reg
        grads['b%d' % self.convL] = db_fc
        loss += softmax_l + reg_W_loss
        if self.use_dropout:
            dropout_cache = self.cache['dropout']
            dout = dropout_backward(dout, dropout_cache)
        for i in range(self.convL - 1,-1,-1):
            reg_W_loss, dW_reg = self.regularization_loss(self.params['W%d' %i])
            loss += reg_W_loss
            grads['W%d' %i] += dW_reg
            dout,dW,db = conv_relu_pool_backward(dout,self.cache['con' + str(i)])
            grads['W%d' % i] += dW
            grads['b%d' % i] = db
            loss += reg_W_loss
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def regularization_loss(self, W):
        loss = self.reg * 0.5 * np.sum(W * W)
        dx = self.reg * W
        return loss, dx
