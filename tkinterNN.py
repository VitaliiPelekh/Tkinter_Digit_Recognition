import tkinter as tk
from tkinter import Button, Label, Canvas
from keras.models import load_model
import numpy as np
from PIL import Image, ImageGrab
import os
from scipy.ndimage.measurements import center_of_mass
import cv2


def getBestShift(img):
    cy, cx = center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


class DigitRecognizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.canvas_width = 300
        self.canvas_height = 300
        self.bg_color = "white"
        self.paint_color = "black"
        self.brush_size = 15
        self.init_ui()
        self.model = load_model('model.h5')

    def init_ui(self):
        self.canvas = Canvas(self, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color)
        self.label = Label(self, text="Draw a digit", font=("Helvetica", 16))
        self.classify_btn = Button(self, text="Classify", command=self.classify_handwriting)
        self.clear_btn = Button(self, text="Clear", command=self.clear_canvas)
        self.result_label = Label(self, text="", font=("Helvetica", 16))
        self.canvas.grid(row=0, column=0, pady=2, padx=2, columnspan=2)
        self.label.grid(row=1, column=0, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.clear_btn.grid(row=1, column=2, pady=2, padx=2)
        self.result_label.grid(row=2, column=0, pady=2, padx=2, columnspan=3)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.paint_color, outline=self.paint_color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="")

    def classify_handwriting(self):
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1))
        image = image.convert('L')  # Конвертуємо в відтінки сірого
        image_np = np.array(image, dtype=np.uint8)  # Перетворюємо PIL Image в NumPy масив

        # Обернення зображення: цифри білі, фон чорний
        image_np = 255 - image_np

        image_np = cv2.resize(image_np, (28, 28), interpolation=cv2.INTER_AREA)

        # Центрування зображення
        shiftx, shifty = getBestShift(image_np)
        shifted = shift(image_np, shiftx, shifty)
        image_np = shifted

        # Нормалізація та зміна форми зображення для моделі
        image_np = image_np.reshape(1, 28, 28, 1)
        image_np = image_np.astype('float32') / 255.0

        # Класифікація зображення
        result = self.model.predict([image_np])[0]
        idx = np.argmax(result)
        self.result_label.config(text=str(idx) + ", " + str(round(max(result) * 100, 2)) + "%")


if __name__ == "__main__":
    app = DigitRecognizer()
    app.title("Digit Recognizer")
    app.mainloop()
