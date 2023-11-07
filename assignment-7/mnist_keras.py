import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model


class MnistKeras:
    def __init__(self):

     self.model=keras.Sequential([
     layers.Dense(100,activation='relu'),
     layers.Dense(10, activation='softmax')
])
     self.x_train=None
     self.x_test=None
     self.y_train=None
     self.y_test=None
    
    def load_data(self):
       (self.x_train, self.y_train) , (self.x_test, self.y_test)=mnist.load_data()

       self.x_train=self.x_train.reshape(60000,28*28)
       self.x_train=self.x_train.astype("float32")/255
       self.x_test=self.x_test.reshape(10000,28*28)
       self.x_test=self.x_test.astype("float32")/255
       return (self.x_train, self.y_train) , (self.x_test, self.y_test)
    
    def build_model(self):
       self.model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=["accuracy"])
       return self.model
    
    def train(self,model,model_name):
       model.fit(self.x_train,self.y_train,epochs=10,batch_size=64)
       self.model.save(model_name)
       return self.model
    
    def load(self,model_name):
       loaded_model=load_model(model_name)
       return loaded_model
    
    def test(self,model_name):
       loaded_model=load_model(model_name)
       loss,accuracy=loaded_model.evaluate(self.x_test,self.y_test)
       print(f'Test accuracy: {accuracy}')
       print(f'Test loss: {loss}')
       
    def predict(self,x):
       pred=self.model.predict(x)
       return pred