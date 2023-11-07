import mnist_keras as mk
import train

m=mk.MnistKeras()
(x_train, y_train) , (x_test, y_test)=m.load_data()
model=m.load("model_your_Tawfic_Shamieh")
m.test(model)