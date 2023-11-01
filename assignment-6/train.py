from mnist_data import MnistData
import numpy as np
import common as c
import pickle
from two_layer_net import TwoLayerNet

# Load the MNIST dataset
mnist_data = MnistData()
(x_train, t_train), (x_test, t_test) = mnist_data.load()

# Initialize the neural network
input_size = 28 * 28  # Assuming 28x28 images
hidden_size = 100  # Adjust as needed
output_size = 10  # Assuming 10 classes for digits 0 to 9
net = TwoLayerNet(input_size, hidden_size, output_size)

# Define hyperparameters
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# Training the neural network
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = net.numerical_gradient(x_batch, t_batch)

    for key in ('w1', 'b1', 'w2', 'b2'):
        net.params[key] -= learning_rate * grads[key]

# Save the trained model as a pickle file
file_name = 'your_Tawfic_Shamieh_mnist_nn_model.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(net, file)
