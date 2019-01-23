import numpy as np
import h5py
from DNN_Classifier import L_layer_model, predict, print_mislabeled_images
from DNN_Utils import load_dataset, process_dataset

# Loading the data (cat and non-cat)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

train_set_x, test_set_x, num_px = process_dataset(train_set_x_orig, test_set_x_orig)

### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

# Train the model
learning_rate = 0.0075
num_iterations = 2500
parameters = L_layer_model(train_set_x, train_set_y, layers_dims, learning_rate, num_iterations, print_cost = True)

# Store the model papameters into a hdf5 file
with h5py.File("mytestfileDNN.h5", "w") as f:
    L = len(layers_dims)
    for l in range(1, L):
        f.create_dataset("W"+str(l), data=parameters["W"+str(l)])
        f.create_dataset("b"+str(l), data=parameters["b"+str(l)])
    f.create_dataset("learning_rate", data = learning_rate)
    f.create_dataset("num_iterations", data = num_iterations)
    f.create_dataset("num_px", data = num_px)
    f.create_dataset("number_of_layers", data = L)
    f.create_dataset("classes", data = classes)
    
pred_train = predict(train_set_x, train_set_y, parameters)
print("Train accuracy: "  + str(np.sum((pred_train == train_set_y)/train_set_x.shape[1])))

pred_test = predict(test_set_x, test_set_y, parameters)
print("Test accuracy: "  + str(np.sum((pred_test == test_set_y)/test_set_x.shape[1])))

print_mislabeled_images(classes, test_set_x, test_set_y, pred_test)

