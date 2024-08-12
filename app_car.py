import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('car_color_model.keras')

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((128, 128))  # Resize to match model input
        img = np.array(img)
        img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input
        img = img / 255.0  # Normalize
        
        # Predict
        prediction = model.predict(img)
        class_idx = np.argmax(prediction, axis=1)
        
        if class_idx == 0:
            result = "Red Car"
        else:
            result = "Blue Car"
        
        result_label.config(text=f"Prediction: {result}")
        img = ImageTk.PhotoImage(Image.open(file_path).resize((200, 200)))
        image_label.config(image=img)
        image_label.image = img

# Setup the GUI
root = tk.Tk()
root.title("Car Color Predictor")

frame = tk.Frame(root)
frame.pack()

load_button = Button(frame, text="Load Image", command=load_image)
load_button.pack()

image_label = Label(frame)
image_label.pack()

result_label = Label(frame, text="Prediction: ")
result_label.pack()

root.mainloop()
