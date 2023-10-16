import sys
from mnist import Mnist

from PIL import Image

def preprocess_image(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    image = image.convert('L')

    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    return image


def main(image_filename, actual_digit):
    # Load your trained model
    model = Mnist()  # You should implement the Mnist class to load your model

    # Load the image using image_filename
    preprocess_image(image_filename)

    # Use the loaded model to make predictions on the image
    predicted_digit = model.predict(image_filename)  # You should implement this method in the Mnist class

    # Compare the predicted digit with the actual_digit
    if predicted_digit == actual_digit:
        print(f"Success: Image {image_filename} is for digit {actual_digit} is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_filename} is for digit {actual_digit} but the inference result is {predicted_digit}.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python module5.py <image_filename> <actual_digit>")
        #sys.exit(1)

    image_filename = sys.argv[1]
    actual_digit = int(sys.argv[2])

    main(image_filename, actual_digit)