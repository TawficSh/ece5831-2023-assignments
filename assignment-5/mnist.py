import gzip
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Mnist:
    img_size=28*28
    model_file_name='model/sample_weight.pkl'
    key_file={
    'test_image': 'mnist/t10k-images-idx3-ubyte.gz',
    'test_label': 'mnist/t10k-labels-idx1-ubyte.gz'}

    def __init__(self):
      self.network=None

    def load_images(self,file_name):
     with gzip.open(file_name, 'rb') as f:
        images=np.frombuffer(f.read(),np.uint8,offset=16)
        
     images=images.reshape(-1,self.img_size)
     print("Done with loading images:",file_name)
     return images

    def load_labels(self,file_name):
     with gzip.open(file_name, 'rb') as f:
        labels=np.frombuffer(f.read(),np.uint8,offset=8)
        
     print("Done with loading labels:",file_name)
     return labels
    
    def sigmoid(self,a):
     return 1/(1+np.exp(-a))
    
    def softmax(self,ss):
     m=np.max(ss)
     a=np.exp(ss-m)
     s=np.sum(a)
     return a/s
    
    def init_network(self,model_file_name):
     with open(self.model_file_name, 'rb') as f:
        self.network=pickle.load(f)
     return self.network
    
    def predict(self,network,x):
     w1,w2,w3=self.network['W1'],self.network["W2"],self.network["W3"]
     b1,b2,b3=self.network['b1'],self.network["b2"],self.network["b3"]
     a1=np.dot(x,w1)+b1
     z1=self.sigmoid(a1)
     a2=np.dot(z1,w2)+b2
     z2=self.sigmoid(a2)
     a3=np.dot(z2,w3)+b3
     y=self.softmax(a3)
     return y