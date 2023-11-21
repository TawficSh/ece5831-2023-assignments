import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
#from keras.models import load_model
from tensorflow.keras.applications import VGG16
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import ModelCheckpoint

class DogsCatsPre:
 def __init__(self):
    self.model=None
 def build_model(self):
    # Load pre-trained VGG16 model without the top (fully connected) layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 180, 3))

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add your custom classification layers
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    self.model = keras.Model(inputs=base_model.input, outputs=outputs)
    return self.model
 
 def train(self,model_name):
   self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
   self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
   data_from_kaggle = "data-from-kaggle/train"
   data_dirname = "dogs-vs-cats"
   batch_size = 32
   train_dataset = image_dataset_from_directory(f"{data_dirname}/train", image_size=(200, 180), batch_size=batch_size)
   validation_dataset = image_dataset_from_directory(f"{data_dirname}/validation", image_size=(200, 180), batch_size=batch_size)
   test_dataset = image_dataset_from_directory(f"{data_dirname}/test", image_size=(200, 180), batch_size=batch_size)
   callbacks = [ keras.callbacks.ModelCheckpoint(
    filepath=model_name,
    save_best_only=False,
    monitor="val_loss"
    )]
   self.model.fit(train_dataset, validation_data=validation_dataset, epochs=10, callbacks=callbacks)


 def predict(self, model_name, file_name):
        img = keras.preprocessing.image.load_img(file_name, target_size=(200, 180))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tensorflow.expand_dims(img_array, 0)
        predictions = self.model.predict(img_array)
        print(predictions)
  