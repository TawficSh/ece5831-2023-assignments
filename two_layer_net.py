from common import sigmoid,softmax,nnumerical_gradient,cross_entropy_error
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        return y


    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {}
        grads['w1'] = nnumerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = nnumerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = nnumerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = nnumerical_gradient(loss_w, self.params['b2'])
        return grads
    

    def forward(self, x):
        self.layer1 = np.dot(x, self.params['w1']) + self.params['b1']
        self.layer_1 = sigmoid(self.layer1)
        self.layer2 = np.dot(self.layer_1, self.params['w2']) + self.params['b2']
        self.y = softmax(self.layer2)
    
    def gradient(self, x, t):
        # Forward pass
        self.forward(x)

        # Backward pass
        grads = {}
        batch_size = x.shape[0]

        # Gradients of weights and biases from the last layer to the output layer
        self.delta2 = (self.y - t) / batch_size
        grads['w2'] = np.dot(self.layer_1.T, self.delta2)
        grads['b2'] = np.sum(self.delta2, axis=0)

        # Gradients of weights and biases from the input layer to the hidden layer
        delta1 = np.dot(self.delta2, self.params['w2'].T) * self.layer_1 * (1 - self.layer_1)
        grads['w1'] = np.dot(x.T, delta1)
        grads['b1'] = np.sum(delta1, axis=0)
        return grads
    

    def accuracy(self, x, t):
        self.forward(x)
        y = np.argmax(self.y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy