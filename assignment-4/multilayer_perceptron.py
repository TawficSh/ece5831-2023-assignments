import numpy as np

#Multi Layer Perceptron Class 

class MultiLayerPerceptron:
    # initialization of the neral network's biases and weights
    def __init__(self):
        self.network={}
        self.network['w1']=np.array([[0.5,0.6,0.7],[0.1,0.2,0.3]])
        self.network['b1']=np.array([0.2,0.3,0.4])
        self.network['w2']=np.array([[0.1,0.2],[0.3,0.4],[0.4,0.5]])
        self.network['b2']=np.array([0.2,0.9])
        self.network['w3']=np.array([[0.5,0.6],[0.2365,0.36598]])
        self.network['b3']=np.array([0.1234,0.9876])

    def sigmoid(self,s):
        #Sigmoid function as an activation function
        return 1/(1+np.exp(-s))
    
    def softmax(self,ss):
        #Softmax function(activation function for the last layer to give probability distribution)
        m=np.max(ss)
        a=np.exp(ss-m)
        s=np.sum(a)
        return a/s
    
    def identity_function(self,s):
        #identity function to return the output
        return s
    
    def forward(self,x):
        #Feed Fprward Neural Network function
        w1,w2,w3=self.network["w1"],self.network["w2"],self.network["w3"]
        b1,b2,b3=self.network["b1"],self.network["b2"],self.network["b3"]

        a1=np.dot(x,w1)+b1
        z1=self.sigmoid(a1)

        a2=np.dot(z1,w2)+b2
        z2=self.sigmoid(a2)

        a3=np.dot(z2,w3)+b3
        z3=self.softmax(a3)

        y=self.identity_function(z3)
        return y