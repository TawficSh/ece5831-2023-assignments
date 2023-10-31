from mnist_data import MnistData 
import numpy as np
import two_layer_net as tln
import train as train
from two_layer_net import net

img1="C:\Users\ACER\OneDrive\Desktop\ece5831-2023-assignments\ece5831-2023-assignments\assignment-5\Images(num)\0_1.jpg"
img2="C:\Users\ACER\OneDrive\Desktop\ece5831-2023-assignments\ece5831-2023-assignments\assignment-5\Images(num)\1_2.jpg"
img3="C:\Users\ACER\OneDrive\Desktop\ece5831-2023-assignments\ece5831-2023-assignments\assignment-5\Images(num)\2_3.jpg"
img4="C:\Users\ACER\OneDrive\Desktop\ece5831-2023-assignments\ece5831-2023-assignments\assignment-5\Images(num)\3_4.jpg"
img5="C:\Users\ACER\OneDrive\Desktop\ece5831-2023-assignments\ece5831-2023-assignments\assignment-5\Images(num)\4_5.jpg"

from PIL import Image
# Load and preprocess the custom test images
# Implement the code to load and preprocess your custom test images

custom_test_image_paths = [img1,img2,img3,img4,img5]
custom_test_images = []
for image_path in custom_test_image_paths:
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize the image to 28x28
    image = np.array(image)  # Convert to a NumPy array
    image = image.reshape(1, 28 * 28)  # Flatten the image to a 1D array
    image = image.astype('float32') / 255  # Normalize pixel values
    custom_test_images.append(image)

# Use the trained model to make predictions on the custom images
# Implement the code to use the trained model for predictions
custom_predictions = []
for image in custom_test_images:
    prediction = net.predict(image)  # Assuming 'net' is the trained neural network
    custom_predictions.append(prediction)

# Display or store the predictions for further analysis
# Implement the code to display or store the predictions
for i, pred in enumerate(custom_predictions):
    print(f"Prediction for {custom_test_image_paths[i]}: {pred}")