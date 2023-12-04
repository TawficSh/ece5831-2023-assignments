from keras import models
from keras import layers
import matplotlib.pyplot as plt

class bostonHousing:
    def __init__(self):
      self.model=None
      self.history=None

    def normalize_data(self,data):
     mean = data.mean(axis=0)
     data -= mean
     std = data.std(axis=0)
     data /= std
     return data

    def build_model(self,x_size):
      self.model = models.Sequential()
      self.model.add(layers.Dense(64, activation='relu', input_shape=(x_size,)))
      self.model.add(layers.Dense(64, activation='relu'))
      self.model.add(layers.Dense(1))
      return  self.model
    
    def train(self,x,y):
      self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
      self.history = self.model.fit(x, y, validation_split=0.2, batch_size=1, epochs=200)
      return self.history

    def predict(self,x):
        predictions=self.model.predict(x)
        return predictions
    
    def visualize(self):
        history_dict = self.history.history
        history_dict.keys()
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, 'r-', label='training loss')
        plt.plot(epochs, val_loss_values, 'b--', label='validation loss')
        plt.title('training vs. validation loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()