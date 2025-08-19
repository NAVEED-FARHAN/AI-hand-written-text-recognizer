import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# Load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Show the first image in the dataset
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.show()

# Print the shape (size) of the dataset
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Normalize the images (convert pixel values from 0-255 to 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Data normalized successfully!")

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras

# Create the Neural Network Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input layer (Flatten 28x28 to 1D)
    keras.layers.Dense(128, activation='relu'),  # Hidden layer (128 neurons, ReLU activation)
    keras.layers.Dense(10, activation='softmax') # Output layer (10 neurons for digits 0-9)
])

# Compile the model (choose optimizer, loss function, and metrics)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model created successfully!")

# Train the model using the training data
model.fit(x_train, y_train, epochs=10)

print("Model training completed!")

# Evaluate the model using test data
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {test_acc * 100:.2f}%")


import matplotlib.pyplot as plt

# Pick a random test image
random_index = np.random.randint(0, len(x_test))  # Choose a random index
test_image = x_test[random_index]  # Get the image
true_label = y_test[random_index]  # Get the actual label

# Show the image
plt.imshow(test_image, cmap='gray')  # Display the image
plt.title(f"Actual Digit: {true_label}")  # Show the actual label
plt.axis('off')  # Hide axes
plt.show()

# Reshape the image for prediction
test_image = test_image.reshape(1, 28, 28)  # Reshape for model input

# Make a prediction
predictions = model.predict(test_image)  # Get predictions
predicted_label = np.argmax(predictions)  # Get the digit with the highest probability

print(f"Model Prediction: {predicted_label}")

model.save("handwriting_model.h5")
