import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model("handwriting_model.h5")  # Ensure you saved your model earlier

# Function to predict the drawn digit
def predict_digit():
    # Save the canvas content as an image
    filename = "digit.png"
    
    # Convert the canvas to an image
    image.save(filename)  # Save as PNG directly

    # Open and preprocess the image
    img = Image.open(filename).convert("L")  # Grayscale
    img = img.resize((28, 28))  # Resize to match the model input

    # Convert image to numpy array
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Display the result
    label.config(text=f"Predicted Digit: {predicted_digit}", font=("Arial", 24))

    # Show the drawn image with the prediction
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digit}")
    plt.axis('off')
    plt.show()

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="black")

# Create GUI window
window = tk.Tk()
window.title("Digit Recognizer")

# Create a canvas for drawing
canvas = Canvas(window, width=280, height=280, bg="black")
canvas.grid(row=0, column=0, columnspan=2)

# Create image to store drawing
image = Image.new("RGB", (280, 280), "black")
draw = ImageDraw.Draw(image)

# Function to draw on canvas
def draw_digit(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x+10, y+10, fill="white", outline="white", width=5)
    draw.ellipse([x, y, x+10, y+10], fill="white")

# Bind mouse events
canvas.bind("<B1-Motion>", draw_digit)

# Create buttons
btn_predict = tk.Button(window, text="Predict", command=predict_digit, font=("Arial", 14))
btn_predict.grid(row=1, column=0)

btn_clear = tk.Button(window, text="Clear", command=clear_canvas, font=("Arial", 14))
btn_clear.grid(row=1, column=1)

# Create label for prediction result
label = tk.Label(window, text="Draw a digit!", font=("Arial", 18))
label.grid(row=2, column=0, columnspan=2)

# Run the application
window.mainloop()

model.save("handwriting_model.h5")

