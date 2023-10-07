import multilayer_perceptron as mlp
import numpy as np

m1=mlp.MultiLayerPerceptron()
y1=m1.forward(np.array([0.1,0.2]))
print(y1)

m2=mlp.MultiLayerPerceptron()
y2=m2.forward(np.array([0.1,0.2]))
print(y2)

m3=mlp.MultiLayerPerceptron()
y3=m3.forward(np.array([8.032,12.3652]))
print(y3)