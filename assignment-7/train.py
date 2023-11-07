import mnist_keras as mk
m=mk.MnistKeras()

(x_train, y_train) , (x_test, y_test)=m.load_data()
x=m.build_model()
m.train(x,"model_your_Tawfic_Shamieh")