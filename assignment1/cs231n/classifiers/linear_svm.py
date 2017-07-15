import numpy as np
from random import shuffle
from numpy import *

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights (C == # of classes == 10)
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # Compute gradients (one inner and one outer sum)
        # Wonderfully compact and hard to read
        dW[y[i],:] -= X[:,i].T # this is really a sum over j != y_i
        dW[j,:] += X[:,i].T # sums each contribution of the x_i's

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Same with gradient
  dW /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)

  # Gradient regularization that carries through per https://piazza.com/class/i37qi08h43qfv?cid=118
  dW += reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  numLoss = 0.0
  dw = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  score = W.dot(X)
  batch = X.shape[1]
  correct_class = score[y, arange(0, batch)]
  delta = 1
  loss = score - correct_class + delta
  loss[loss < 0] = 0
  loss[y, arange(0, batch)] = 0
  numLoss = sum(loss) / batch + 0.5 * reg * sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  loss[loss > 0] = 1
  loss[y, arange(0, batch)] = -sum(loss, axis=0)
  dw = loss.dot(X.T);
  dw /= batch;
  dw += reg * (W);
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return numLoss, dw
