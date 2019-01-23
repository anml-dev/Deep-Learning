import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(Z):
    """
    Implements the sigmoid activation
    Z -- numpy array of any shape    
    """
    
    A = 1/(1+np.exp(-Z))
    
    return A

def relu(Z):
    """
    Implement the RELU function.
    Z -- Output of the linear layer, of any shape
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    return A


def initialize_parameters(layer_dims):
    """
    Initialize parameters - random initialization for the weight matrices, and zero initialization for the biases.
    
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    parameters -- python dictionary containing the parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])/np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    
    AL -- last post-activation value; probability vector corresponding to our label predictions, shape (1, number of examples)
    caches -- list containing the caches from all layers
    """

    caches = []
    A = X
    L = len(parameters) // 2   # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        linear_cache = (A_prev, W, b)
        
        Z = W.dot(A_prev) + b
        activation_cache = Z
        
        A = relu(Z)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        
        cache = (linear_cache, activation_cache)
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    A_prev = A 
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    linear_cache = (A_prev, W, b)
        
    Z = W.dot(A_prev) + b
    activation_cache = Z
        
    AL = sigmoid(Z)
    assert(AL.shape == (1, X.shape[1]))
    #assert (AL.shape == (W.shape[0], A_prev.shape[1]))
    
    cache = (linear_cache, activation_cache)
    caches.append(cache) 
            
    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function.

    AL -- probability vector corresponding to our label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from AL and y.
    cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost) 
    assert(cost.shape == ())
    
    return cost

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for a single layer.
    
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache), stored for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    Z = activation_cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    if activation == "relu":
        dZ = np.array(dA, copy=True) # converting dz to a correct object. 
        dZ[Z <= 0] = 0  # When z <= 0, we should set dz to 0 as well.
        assert (dZ.shape == Z.shape)
        
    elif activation == "sigmoid":
        s = sigmoid(Z)
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)

    dW = 1./m * np.dot(dZ, A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for all the layers --> [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID
    
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list containing the caches from all layers
                    
    Returns a dictionary with the gradients.
    """
    grads = {}
    L = len(caches) # the number of layers
    
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a L-layer neural network.
    """
    
    m = X.shape[1]
    p = np.zeros((1,m))
    
    # Forward propagation
    AL, caches = L_model_forward(X, parameters)

    # Convert the probability vector AL (output of the forward propagation) to 0/1 predictions
    for i in range(0, AL.shape[1]):
        if AL[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
       
    return p

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
    plt.show()    


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    """

    #np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
       
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
         
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
