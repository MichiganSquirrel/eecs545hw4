import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
               hidden_dim=100, num_classes=10, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters_1: Number of filters to use in the first convolutional layer
        - num_filters_2: Number of filters to use in the second convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        (self.C, self.H, self.W) = input_dim
        self.filter_size = filter_size
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Your initializations should work for any valid input dims,      #
        # number of filters, hidden dims, and num_classes. Assume that we use      #
        # max pooling with pool height and width 2 with stride 2.                  #
        #                                                                          #
        # For Linear layers, weights and biases should be initialized from a       #
        # uniform distribution from -sqrt(k) to sqrt(k),                           #
        # where k = 1 / (#input features)                                          #
        # For Conv. layers, weights should be initialized from a uniform           #
        # distribution from -sqrt(k) to sqrt(k),                                   #
        # where k = 1 / ((#input channels) * filter_size^2)                        #
        # Note: we use the same initialization as pytorch.                         #
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html           #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html           #
        #                                                                          #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights for the convolutional layer using the keys 'W1' and 'W2'   #
        # (here we do not consider the bias term in the convolutional layer);      #
        # use keys 'W3' and 'b3' for the weights and biases of the                 #
        # hidden fully-connected layer, and keys 'W4' and 'b4' for the weights     #
        # and biases of the output affine layer.                                   #
        #                                                                          #
        # Make sure you have initialized W1, W2, W3, W4, b3, and b4 in the         #
        # params dicitionary.                                                      #
        #                                                                          #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3. Calculate the size of W3 dynamically           #
        ############################################################################
        #! first conv layer
        k = 1.0/((self.C) * (self.filter_size**2))
        W1 = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k),
                               size=(num_filters_1, self.C, filter_size, filter_size))
        # Weight shape: (num_filters_1, C, filter_size, filter_size)
        self.params['W1'] = W1

        #! output shape after conv layer 1
        H1 = (self.H - filter_size + 1)
        W1 = (self.W - filter_size + 1)

        #! after max pool layer 1
        H1_pool = H1 // 2
        W1_pool = W1 // 2

        #! second conv layer
        k = 1.0/((num_filters_1) * (self.filter_size**2))
        W2 = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k),
                               size=(num_filters_2, num_filters_1, filter_size, filter_size))
        self.params['W2'] = W2
        

        # conv2 output size (valid convolution):
        # H2 = H1_pool - filter_size + 1,  W2_dim = W1_pool - filter_size + 1
        H2 = H1_pool - filter_size + 1
        W2_dim = W1_pool - filter_size + 1

        # After 2x2 max pool with stride 2:
        H2_pool = H2 // 2
        W2_pool = W2_dim // 2

        #! fully connected layer
        flat_dim = num_filters_2 * H2_pool * W2_pool
        W3 = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k),
                               size=(flat_dim, hidden_dim))
        b3 = np.zeros(hidden_dim)
        self.params['W3'] = W3
        self.params['b3'] = b3

        #! output layer
        k = 1.0 / hidden_dim
        W4 = np.random.uniform(low=-np.sqrt(k), high=np.sqrt(k),
                               size=(hidden_dim, num_classes))
        b4 = np.zeros(num_classes)
        self.params['W4'] = W4
        self.params['b4'] = b4

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3.                                                #
        ############################################################################
        #! implement the forward pass
        #! conv1
        conv1_out, cache_conv1 = conv_forward(X, W1)
        relu1_out, cache_relu1 = relu_forward(conv1_out)
        pool1_out, cache_pool1 = max_pool_forward(relu1_out, pool_param)

        #! conv2
        conv2_out, cache_conv2 = conv_forward(pool1_out, W2)
        relu2_out, cache_relu2 = relu_forward(conv2_out)
        pool2_out, cache_pool2 = max_pool_forward(relu2_out, pool_param)

        #! dim = (N, num_filters_2, H2_pool, W2_pool)
        #! flatten the output of the second max pool layer
        N = pool2_out.shape[0]   # 从 pool2_out 的第一维直接取样本数
        flat_dim = np.prod(pool2_out.shape[1:])
        flat_out = pool2_out.reshape(N, flat_dim)

        ## 4. 第一全连接层 (fc1): 输出形状 (N, hidden_dim)
        fc1_out, cache_fc1 = fc_forward(flat_out, W3, b3)
        relu3_out, cache_relu3 = relu_forward(fc1_out)

        ## 5. 第二全连接层 (fc2): 输出形状 (N, num_classes)
        scores, cache_fc2 = fc_forward(relu3_out, W4, b4)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k].                                                      #
        # Hint: The backwards from W3 needs to be un-flattened before it can be    #
        # passed into the max pool backwards                                       #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        d_relu3, dW4, db4 = fc_backward(dscores, cache_fc2)
        grads['W4'] = dW4
        grads['b4'] = db4

        #! relu3 backward
        d_fc1 = relu_backward(d_relu3, cache_relu3)
        d_flat, dW3, db3 = fc_backward(d_fc1, cache_fc1)
        grads['W3'] = dW3
        grads['b3'] = db3

        d_pool2 = d_flat.reshape(pool2_out.shape)
        d_relu2 = max_pool_backward(d_pool2, cache_pool2)
        d_conv2 = relu_backward(d_relu2, cache_relu2)

        d_pool1, dW2 = conv_backward(d_conv2, cache_conv2)
        grads['W2'] = dW2

        d_relu1 = max_pool_backward(d_pool1, cache_pool1)
        d_conv1 = relu_backward(d_relu1, cache_relu1)
        dx, dW1 = conv_backward(d_conv1, cache_conv1)
        grads['W1'] = dW1

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
