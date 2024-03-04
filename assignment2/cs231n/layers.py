from builtins import range
from logging import WARN
from types import GetSetDescriptorType
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_reshaped = np.reshape(x, (x.shape[0], -1))
    out = x_reshaped.dot(w) + b.reshape(b.shape[0])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout.dot(w.T)
    dx = np.reshape(dx, x.shape)

    x = np.reshape(x, (x.shape[0], -1))
    dw = x.T.dot(dout)

    db = np.sum(dout, axis=0)
    db = np.reshape(db, b.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mask = x > 0
    dx = mask * dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = x.shape[0]
    num_classes = x.shape[1]
    
    # vectorized forward pass using staged computation
    scores = x
    scores = scores - np.max(scores, axis=1, keepdims=True)
    num = np.exp(scores)
    den = num.sum(axis=1, keepdims=True)
    invden = 1.0 / den
    probs = num * invden
    loss = np.sum(-np.log(probs[np.arange(num_train), y]))

    # vectorized backward pass using staged computation
    dprobs = np.zeros(probs.shape)
    dprobs[np.arange(num_train), y] = -1.0 / probs[np.arange(num_train), y]
    dinvden = np.sum(num * dprobs, axis=1, keepdims=True)
    dnum = invden * dprobs
    dden = np.sum((-1.0 / (den ** 2)) * dinvden, axis=1, keepdims=True)
    dnum += np.ones(num.shape) * dden
    dscores = num * dnum

    # entire batch average
    loss /= num_train
    dx = dscores/num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mu = np.sum(x, axis=0, keepdims=True) / N
        diff = x - mu
        diffsqr = diff ** 2
        var = np.sum(diffsqr, axis=0, keepdims=True) / N
        std = np.sqrt(var + eps)
        invstd = 1 / std
        xhat = diff * invstd
        out = gamma * xhat + beta
        
        cache = (xhat, gamma, std, invstd, diff, var, eps, N)
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        xhat = (x - running_mean) / (running_var + eps)**(1/2)
        out = gamma * xhat + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    xhat, gamma, std, invstd, diff, var, eps, N = cache
    dgamma = np.sum(xhat * dout, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)
    dxhat = gamma * dout
    ddiff = dxhat * invstd
    dinvstd = np.sum(diff * dxhat, axis=0, keepdims=True)
    dstd = (-1 / std**2) * dinvstd
    dvar = 1 / (2 * np.sqrt(var + eps)) * dstd
    ddiffsqr = np.ones(diff.shape) * dvar / N
    ddiff += 2 * diff * ddiffsqr
    dx = ddiff
    dmu = -np.sum(ddiff, axis=0, keepdims=True)
    dx += np.ones(xhat.shape) * dmu / N
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    xhat, gamma, std, invstd, diff, var, eps, N = cache
    dgamma = np.sum(xhat * dout, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)
    dxhat = gamma * dout
    dx = (1 / std) * np.ones(xhat.shape) * dxhat
    dmu = np.sum(-dxhat / std, axis=0, keepdims=True)
    dvar = np.sum(diff * dxhat, axis = 0, keepdims=True) * (-0.5) * (var + eps) ** (-3/2)
    dx += (1 / N) * 2 * diff * dvar
    dmu += (1/ N) * np.sum(-2 * diff, axis=0, keepdims=True) * dvar
    dx += (1 / N) * np.ones(xhat.shape) * dmu

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = x.shape
    mu = np.sum(x, axis=1, keepdims=True) / D
    diff = x - mu
    diffsqr = diff ** 2
    var = np.sum(diffsqr, axis=1, keepdims=True) / D
    std = np.sqrt(var + eps)
    invstd = 1 / std
    xhat = diff * invstd
    out = gamma * xhat + beta
    cache = (xhat, gamma, std, invstd, diff, var, eps, D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    xhat, gamma, std, invstd, diff, var, eps, D = cache
    dgamma = np.sum(xhat * dout, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)
    dxhat = gamma * dout
    ddiff = dxhat * invstd
    dinvstd = np.sum(diff * dxhat, axis=1, keepdims=True)
    dstd = (-1 / std**2) * dinvstd
    dvar = 1 / (2 * np.sqrt(var + eps)) * dstd
    ddiffsqr = np.ones(diff.shape) * dvar / D
    ddiff += 2 * diff * ddiffsqr
    dx = ddiff
    dmu = -np.sum(ddiff, axis=1, keepdims=True)
    dx += np.ones(xhat.shape) * dmu / D

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = np.random.rand(*x.shape) < p
        out = (mask * x) / p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        p = dropout_param["p"]
        dx = dout * mask / p

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    p, s = conv_param['pad'], conv_param['stride']
    H2 = int(1 + (H + 2 * p - HH) / s)
    W2 = int(1 + (W + 2 * p - WW) / s)
    out = np.zeros((N, F, H2, W2))
    pad_width = ((0, 0), (0, 0), (p, p), (p, p))
    x_pad = np.pad(x, pad_width=pad_width)
    for n in range(N):
        for f in range(F):
            for i in range(H2):
                for j in range(W2):
                    x_receptive = x_pad[n, :, i*s:i*s+HH, j*s:j*s+WW]
                    dot_prod = np.sum(x_receptive * w[f]) + b[f]
                    out[n, f, i, j] = dot_prod

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    p, s = conv_param['pad'], conv_param['stride']
    H2 = int(1 + (H + 2 * p - HH) / s)
    W2 = int(1 + (W + 2 * p - WW) / s)
    out = np.zeros((N, F, H2, W2))
    pad_width = ((0, 0), (0, 0), (p, p), (p, p))
    x_pad = np.pad(x, pad_width=pad_width)
    dx_pad = np.zeros(x_pad.shape)

    for n in range(N):
        for f in range(F):
            for i in range(H2):
                for j in range(W2):
                    x_receptive = x_pad[n, :, i*s:i*s+HH, j*s:j*s+WW]
                    dw[f] += (x_receptive * dout[n, f, i, j])
                    db[f] += dout[n, f, i, j]
                    dx_pad[n, :, i*s:i*s+HH, j*s:j*s+WW] += (w[f] * dout[n, f, i, j])

    dx = dx_pad[:, :, p:-p, p:-p]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


# <--------------------------------------------------------------------------> #

def conv_forward_im2col_naive(x, w, b, conv_param):
    """Im2col implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param, x_col, w_row, out_col)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    p, s = conv_param['pad'], conv_param['stride']
    H2 = int(1 + (H + 2 * p - HH) / s)
    W2 = int(1 + (W + 2 * p - WW) / s)
    out = np.zeros((N, F, H2, W2))
    pad_width = ((0, 0), (0, 0), (p, p), (p, p))
    x_pad = np.pad(x, pad_width=pad_width)
    x_col = np.zeros((N, HH*WW*C, H2*W2))
    out_col = np.zeros((N, F, H2*W2))

    count = 0
    for n in range(N):
        for i in range(H2):
            for j in range(W2):
                x_receptive = x_pad[n, :, i*s:i*s+HH, j*s:j*s+WW]
                x_col[n, :, count] = x_receptive.flatten()
                count += 1
        count = 0

    w_row = np.reshape(w, (F, HH*WW*C))
    for n in range(N):
        out_col[n, :, :] = np.dot(w_row, x_col[n, :, :]) + b.reshape(-1, 1)

    out = np.reshape(out_col, out.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param, x_col, w_row, out_col)
    return out, cache


def conv_backward_im2col_naive(dout, cache):
    """Im2col implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param, x_col, w_row, out_col)
             as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param, x_col, w_row, out_col = cache
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    p, s = conv_param['pad'], conv_param['stride']
    H2 = int(1 + (H + 2 * p - HH) / s)
    W2 = int(1 + (W + 2 * p - WW) / s)
    out = np.zeros((N, F, H2, W2))
    pad_width = ((0, 0), (0, 0), (p, p), (p, p))
    x_pad = np.pad(x, pad_width=pad_width)

    dx_col = np.zeros(x_col.shape)
    dw_row = np.zeros(w_row.shape)
    dout_col = np.reshape(dout, out_col.shape)
    dx_pad = np.zeros(x_pad.shape)

    for n in range(N):
        dw_row += np.dot(dout_col[n], x_col[n, :, :].T)
        dx_col[n, :, :] += np.dot(w_row.T, dout_col[n])
        db += np.sum(dout_col[n], axis=1)

    # inverting the im2col forward operation to get dx
    count = 0
    for n in range(N):
        for i in range(H2):
            for j in range(W2):
                dx_receptive = np.reshape(dx_col[n, :, count], (C, HH, WW)) 
                dx_pad[n, :, i*s:i*s+HH, j*s:j*s+WW] += dx_receptive
                count += 1
        count = 0

    dx = dx_pad[:, :, p:-p, p:-p]
    dw = np.reshape(dw_row, w.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def conv_forward_im2col_naive_fast(x, w, b, conv_param):
    """Im2col (batch) implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param, x_col, w_row, out_col)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    p, s = conv_param['pad'], conv_param['stride']
    H2 = int(1 + (H + 2 * p - HH) / s)
    W2 = int(1 + (W + 2 * p - WW) / s)
    out = np.zeros((N, F, H2, W2))
    pad_width = ((0, 0), (0, 0), (p, p), (p, p))
    x_pad = np.pad(x, pad_width=pad_width)
    x_col = np.zeros((HH*WW*C, H2*W2*N))

    count = 0
    for n in range(N):
        for i in range(H2):
            for j in range(W2):
                x_receptive = x_pad[n, :, i*s:i*s+HH, j*s:j*s+WW]
                x_col[:, count] = x_receptive.flatten()                
                count += 1

    w_row = np.reshape(w, (F, HH*WW*C))
    out_col = np.dot(w_row, x_col) + b.reshape(-1, 1)
    out = out_col.reshape(F, N, H2, W2)
    out = out.transpose(1, 0, 2, 3)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param, x_col, w_row, out_col)
    return out, cache


def conv_backward_im2col_naive_fast(dout, cache):
    """Im2col (batch) implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param, x_col, w_row, out_col)
             as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param, x_col, w_row, out_col = cache
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    p, s = conv_param['pad'], conv_param['stride']
    H2 = int(1 + (H + 2 * p - HH) / s)
    W2 = int(1 + (W + 2 * p - WW) / s)
    out = np.zeros((N, F, H2, W2))
    pad_width = ((0, 0), (0, 0), (p, p), (p, p))
    x_pad = np.pad(x, pad_width=pad_width)

    dout_col = dout.transpose(1, 0, 2, 3)
    dout_col = np.reshape(dout_col, out_col.shape)
    dx_pad = np.zeros(x_pad.shape)

    dw_row = np.dot(dout_col, x_col.T)
    dx_col = np.dot(w_row.T, dout_col)
    db = np.sum(dout_col, axis=1)

    # inverting the im2col forward operation to get dx
    count = 0
    for n in range(N):
        for i in range(H2):
            for j in range(W2):
                dx_receptive = np.reshape(dx_col[:, count], (C, HH, WW)) 
                dx_pad[n, :, i*s:i*s+HH, j*s:j*s+WW] += dx_receptive
                count += 1

    dx = dx_pad[:, :, p:-p, p:-p]
    dw = np.reshape(dw_row, w.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

# <--------------------------------------------------------------------------> #


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    s = pool_param['stride']
    H2 = int(1 + (H - ph) / s)
    W2 = int(1 + (W - pw) / s)
    out = np.zeros((N, C, H2, W2))

    for i in range(H2):
        for j in range(W2):
            x_receptive = x[:, :, i*s:i*s+ph, j*s:j*s+pw]
            max_val = np.max(x_receptive, axis=(2, 3))
            out[:, :, i, j] = max_val

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    s = pool_param['stride']
    H2 = int(1 + (H - ph) / s)
    W2 = int(1 + (W - pw) / s)
    dx = np.zeros(x.shape)

    for i in range(H2):
        for j in range(W2):
            x_receptive = x[:, :, i*s:i*s+ph, j*s:j*s+pw]
            x_map = x_receptive == np.max(x_receptive, axis=(2, 3), keepdims=True)
            dx_receptive = x_map * dout[:, :, i, j].reshape(N, C, 1, 1)
            dx[:, :, i*s:i*s+ph, j*s:j*s+pw] += dx_receptive

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    momentum = bn_param.get('momentum', 0.9)
    eps = bn_param.get('eps', 1e-5)
    running_mean = bn_param.get('running_mean', np.zeros(N * H * W))
    running_var = bn_param.get('running_var', np.zeros(N * H * W))

    mu = np.sum(x, axis=(0, 2, 3), keepdims=True) / (N * H * W)
    diff = x - mu
    diffsqr = diff ** 2
    var = np.sum(diffsqr, axis=(0, 2, 3), keepdims=True) / (N * H * W)
    std = np.sqrt(var + eps)
    invstd = 1 / std
    xhat = diff * invstd
    gamma = gamma.reshape((1, C, 1, 1))
    beta = beta.reshape((1, C, 1, 1))
    out = gamma * xhat + beta
    
    cache = (xhat, gamma, std, invstd, diff, var, eps, N, C, H, W)
    running_mean = momentum * running_mean + (1 - momentum) * mu
    running_var = momentum * running_var + (1 - momentum) * var

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    xhat, gamma, std, invstd, diff, var, eps, N, C, H, W = cache
    dgamma = np.sum(xhat * dout, axis=(0, 2, 3))
    dbeta = np.sum(dout, axis=(0, 2, 3))
    dxhat = gamma * dout
    ddiff = dxhat * invstd
    dinvstd = np.sum(diff * dxhat, axis=(0, 2, 3), keepdims=True)
    dstd = (-1 / std**2) * dinvstd
    dvar = 1 / (2 * np.sqrt(var + eps)) * dstd
    ddiffsqr = np.ones(diff.shape) * dvar / (N * H * W)
    ddiff += 2 * diff * ddiffsqr
    dx = ddiff
    dmu = -np.sum(ddiff, axis=(0, 2, 3), keepdims=True)
    dx += np.ones(xhat.shape) * dmu / (N * H * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    # can be done in one step as well
    x_group = np.reshape(x, (N, G, C//G, H, W))
    x_group2d = np.reshape(x_group, (N*G, -1))

    NG, D = x_group2d.shape 
    mu = np.sum(x_group2d, axis=1, keepdims=True) / D
    diff = x_group2d - mu
    diffsqr = diff ** 2
    var = np.sum(diffsqr, axis=1, keepdims=True) / D
    std = np.sqrt(var + eps)
    invstd = 1 / std
    xhat = diff * invstd
    out = gamma * xhat.reshape(x.shape) + beta
    cache = (xhat, gamma, std, invstd, diff, var, eps, D, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    xhat, gamma, std, invstd, diff, var, eps, D, G = cache
    dgamma = np.sum(xhat.reshape(dout.shape) * dout, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dxhat = gamma * dout
    dxhat = dxhat.reshape(xhat.shape)
    ddiff = dxhat * invstd
    dinvstd = np.sum(diff * dxhat, axis=1, keepdims=True)
    dstd = (-1 / std**2) * dinvstd
    dvar = 1 / (2 * np.sqrt(var + eps)) * dstd
    ddiffsqr = np.ones(diff.shape) * dvar / D
    ddiff += 2 * diff * ddiffsqr
    dx = ddiff
    dmu = -np.sum(ddiff, axis=1, keepdims=True)
    dx += np.ones(xhat.shape) * dmu / D
    dx = dx.reshape(dout.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
