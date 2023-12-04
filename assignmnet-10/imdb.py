import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import matplotlib.pyplot as plt

class IMDB:
    def __init__(self):
     self.NUM_WORDS=10000
     self.NUM_EPOCHS=30
     self.BATCH_SIZE=512
     self.VALIDATION_SPLIT=0.2
     self.PATIENCE=3
     self.model=None
     self.history=None

    def vectorize_sequences(self,sequences):
     results = np.zeros((len(sequences),self.NUM_WORDS))
     for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
     return results

    def build_model(self):
      self.model = models.Sequential()
      self.model.add(layers.Dense(16, activation='relu', input_shape=(self.NUM_WORDS,)))
      self.model.add(layers.Dense(16, activation='relu'))
      self.model.add(layers.Dense(1, activation="sigmoid"))
      return self.model
    
    def train(self,x,y):
      self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.PATIENCE)
      self.history = self.model.fit(x, y, epochs=self.NUM_EPOCHS, 
                    batch_size=self.BATCH_SIZE, validation_split=self.VALIDATION_SPLIT, 
                    callbacks=[callback])
      return self.history
    
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

    def save_model(self,name):
        self.model.save(name)
        
    def predict(self,x):
        predictions=self.model.predict(x)
        return predictions