import numpy as np
import h5py
import matplotlib.pyplot as plt
import skimage
from DNN_Classifier import predict


parameters = {}

with h5py.File('mytestfileDNN.h5', 'r') as f:
   num_px = f['/num_px'].value
   classes = np.array(f['/classes'][:])
   L = f['/number_of_layers'].value
   for l in range(1, L):
      parameters["W" + str(l)] = np.array(f["/W" + str(l)][:])
      parameters["b" + str(l)] = np.array(f["/b" + str(l)][:])


# Test with your own image 

my_image = "Vito.jpeg"    
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
#my_image = "Hydrangeas.jpg"
#my_label_y = [0] 

fname = "images/" + my_image
image = np.array(plt.imread(fname))
my_image = skimage.transform.resize(image, (num_px, num_px), mode='constant', anti_aliasing='None').reshape((num_px*num_px*3, 1))

my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
plt.show()
print ("\ny = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.\n")
