import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# Load the model
model = tf.keras.models.load_model('final_model.keras')  # or 'final_model.keras'

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to predict and display results
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = preprocess_image(file_path)
        
        # Perform prediction
        color_pred, car_count_pred, male_count_pred, female_count_pred = model.predict(img)
        
        # Get the predicted color
        color_classes = ['Red', 'Blue', 'Green', 'Yellow']  # Update with actual classes
        predicted_color = color_classes[np.argmax(color_pred)]
        
        # Display the results
        result_label.config(text=f"Predicted Color: {predicted_color}\n"
                                 f"Car Count: {int(car_count_pred[0][0])}\n"
                                 f"Male Count: {int(male_count_pred[0][0])}\n"
                                 f"Female Count: {int(female_count_pred[0][0])}")
        
        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Initialize the GUI application
root = tk.Tk()
root.title("Car Detection and Counting")

# Button to select an image file
select_button = tk.Button(root, text="Select Image", command=predict_image)
select_button.pack()

# Label to display the selected image
image_label = Label(root)
image_label.pack()

# Label to display the prediction results
result_label = Label(root, text="Predicted results will appear here.", justify=tk.LEFT)
result_label.pack()

# Run the GUI loop
root.mainloop()
