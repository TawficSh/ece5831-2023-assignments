from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

class ReutersClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = Tokenizer(num_words=10000)

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
        x_train = self.tokenizer.sequences_to_matrix(x_train, mode='binary')
        x_test = self.tokenizer.sequences_to_matrix(x_test, mode='binary')
        num_classes = max(y_train) + 1
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        return x_train, y_train, x_test, y_test

    def build_model(self, input_dim, output_dim):
        model = Sequential()
        model.add(Dense(512, input_shape=(input_dim,), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model = model

    def train_model(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        self.model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=validation_data)

    def predict_examples(self, x_test, y_test, num_examples=5):
        predictions = self.model.predict(x_test[:num_examples])

        for i in range(num_examples):
            print(f"Example {i + 1}:")
            print(f"Predicted probabilities: {predictions[i]}")
            predicted_label = predictions[i].argmax()
            print(f"Predicted label index: {predicted_label}")
            print(f"Actual label index: {y_test[i].argmax()}")
            print("-----")
