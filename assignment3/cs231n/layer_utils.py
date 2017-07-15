from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that performs an affine transform, batch normalization,
  and ReLU.

  Inputs:
  - x: Array of shape (N, D1); input to the affine layer
  - w, b: Arrays of shape (D2, D2) and (D2,) giving the weight and bias for
    the affine transform.
  - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
    parameters for batch normalization.
  - bn_param: Dictionary of parameters for batch normalization.

  Returns:
  - out: Output from ReLU, of shape (N, D2)
  - cache: Object to give to the backward pass.
  """
  a, fc_cache = affine_forward(x, w, b)
  a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(a_bn)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine-batchnorm-relu convenience layer.
  """
  fc_cache, bn_cache, relu_cache = cache
  da_bn = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = batchnorm_backward(da_bn, bn_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta  


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def conv_bn_relu_pooling_forward(x, w, b, gamma, beta, conv_param, bn_param,pool_param):
  out0,cache0 = conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param)
  out1, pool_cache = max_pool_forward_fast(out0, pool_param)
  cache = (cache0[0], cache0[1], cache0[2], pool_cache)
  return out1, cache

def conv_bn_relu_pooling_backward(dout, cache):
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  dout = max_pool_backward_fast(dout,pool_cache)
  dan = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta

def deconv_forward(x,w,b,conv_param):
  # deconv need x, w, b, conv_param, x_cols = cache
  p = conv_param['pad']
  yn = x.shape[2] - 2 * p + w.shape[2] - 1
  y = np.zeros((x.shape[0], w.shape[1], yn, yn))
  # N, C, H, W = x.shape
  # F, c, HH, WW = w.shape
  deconvcache = prepare_conv_cache(y,w,b,conv_param)
  x0,_,_ = conv_backward_fast(x,deconvcache)
  cache = (x, deconvcache[1],deconvcache[2],deconvcache[3])
  #dx, dw, db
  return x0,cache

def transposedConv_forward(x,w,b,conv_param):
  # http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#no-zero-padding-unit-strides-transposed
  out, cache = conv_forward_fast(x,w,b,conv_param)
  return out,cache

def deconv_bn_relu_unpooling_forward(x, w, b, gamma, beta, pool_cache,conv_param,bn_param):
  _,_,_,pcache = pool_cache
  #pcache = pool_cache
  x,deconvcache = transposedConv_forward(x,w,b,conv_param)
  x, bn_cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
  x, relu_cache = relu_forward(x)
  x = max_pool_backward_fast(x,pcache)
  cache = (deconvcache,bn_cache, relu_cache,pcache)
  return x,cache

def deconv_backward(dout2,deconvcache):
  low, w, b, conv_param = deconvcache
  dout3, dxconv_cache = conv_forward_fast(dout2, w, b, conv_param)
  _, dw, db = conv_backward_fast(low, dxconv_cache)
  return dout3,dw,db

def transposedConv_backward(dout,cache):
  # http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#no-zero-padding-unit-strides-transposed
  dx, dw, db = conv_backward_fast(dout,cache)
  return dx, dw, db

def deconv_bn_relu_unpooling_backward(dout, cache):
  deconvcache,bn_cache, relu_cache,pool_cache = cache
  dout = max_upPooling_backward_reshape(dout,pool_cache)
  dout = relu_backward(dout,relu_cache)
  dout, dgamma, dbeta = spatial_batchnorm_backward(dout,bn_cache)
  dout,dw,db = transposedConv_backward(dout,deconvcache)
  return dout,dw,db,dgamma, dbeta

def deconv_bn_relu_forward(x, w, b, gamma, beta,conv_param,bn_param):
  x, deconvcache = transposedConv_forward(x, w, b, conv_param)
  x, bn_cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
  x, relu_cache = relu_forward(x)
  cache = (deconvcache, bn_cache, relu_cache)
  return x, cache

def deconv_bn_relu_backward(dout, cache):
  deconvcache,bn_cache, relu_cache = cache
  dout = relu_backward(dout,relu_cache)
  dout, dgamma, dbeta = spatial_batchnorm_backward(dout,bn_cache)
  dout,dw,db = transposedConv_backward(dout,deconvcache)
  return dout,dw,db,dgamma, dbeta

def shortcut_deconv_bn_relu_forward(x,shortCutX, w, b, gamma, beta,conv_param,bn_param):
  x = np.concatenate((x,shortCutX),axis=1)
  return deconv_bn_relu_forward(x,w, b, gamma, beta,conv_param,bn_param)

def shortcut_deconv_bn_relu_backward(dout, cache):
  dout, dw, db, dgamma, dbeta = deconv_bn_relu_backward(dout, cache)
  _,c,_,_ = dout.shape
  return dout[:,:c / 2,:,:],dout[:,c / 2:,:,:], dw, db, dgamma, dbeta

def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(an)
  cache = (conv_cache, bn_cache, relu_cache)
  return out, cache

def conv_bn_relu_backward(dout, cache):
  conv_cache, bn_cache, relu_cache = cache
  dan = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta

def conv_bn_relu_byway_forward(x, w, b, gamma, beta, conv_param, bn_param):
  out,cache = conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param)
  return out, cache, x

def conv_bn_relu_byway_backward(dout, cache, dout1):
  dx, dw, db, dgamma, dbeta = conv_bn_relu_backward(dout, cache)

  return out, cache

def conv_bn_relu_conv_bn_byway_relu_forward(x,wb,gb,conv_param,bn_param):
  w0,b0,w1,b1 = wb
  gamma0,beta0,gamma1,beta1 = gb
  bn_param0, bn_param1 = bn_param
  a0, conv_cache0 = conv_forward_fast(x, w0, b0, conv_param)
  an0, bn_cache0 = spatial_batchnorm_forward(a0, gamma0, beta0, bn_param0)
  out0, relu_cache0 = relu_forward(an0)
  a1, conv_cache1 = conv_forward_fast(out0, w1, b1, conv_param)
  an1, bn_cache1 = spatial_batchnorm_forward(a1, gamma1, beta1, bn_param1)
  xcopy = np.zeros_like(an1)
  xcopy[:,:x.shape[1],:,:] = x
  temp = an1 + xcopy
  out1, relu_cache1 = relu_forward(temp)
  cache = (conv_cache0, bn_cache0, relu_cache0,conv_cache1,bn_cache1,relu_cache1)
  return out1, cache

def conv_bn_relu_conv_bn_byway_relu_backward(dout,cache):
  conv_cache0, bn_cache0, relu_cache0, conv_cache1, bn_cache1, relu_cache1 = cache
  dtemp = relu_backward(dout, relu_cache1)
  dan1 = dtemp
  da1, dgamma1, dbeta1 = spatial_batchnorm_backward(dan1,bn_cache1)
  dx1, dw1, db1 = conv_backward_fast(da1, conv_cache1)
  dan0 = relu_backward(dx1,relu_cache0)
  da0, dgamma0, dbeta0 = spatial_batchnorm_backward(dan0, bn_cache0)
  dx0, dw0, db0 = conv_backward_fast(da0, conv_cache0)
  dx0 += dtemp[:,:dx0.shape[1],:,:]
  dwb = (dw0,db0,dw1,db1)
  dgb = (dgamma0,dbeta0,dgamma1,dbeta1)
  return dx0, dwb, dgb

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db



