from mnist import MNIST
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from tkinter import messagebox
import numpy as np
class DrawingApp:
    def __init__(self, neural_network):
        self.window = tk.Tk()
        self.window.title("Desenhe um Número")
        self.canvas = tk.Canvas(self.window, width=280, height=280, bg="black")
        self.canvas.pack(pady=20)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.process_button = tk.Button(self.window, text="Processar", command=self.process_image)
        self.process_button.pack()

        self.image = Image.new("L", (280, 280), "black")
        self.draw_image = ImageDraw.Draw(self.image)
        self.nn = neural_network
        self.clear_button = tk.Button(self.window, text="Limpar", command=self.clear_canvas)
        self.clear_button.pack(pady=10)

    def draw(self, event):
        x, y = event.x, event.y
        r = 5  # Raio do ponto
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill=255, outline=255)

    def process_image(self):
        # Resize to 28x28 pixels
        image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        image_array = np.array(image)

        coords = np.column_stack(np.where(image_array > 0))
        if coords.size == 0:
            messagebox.showinfo("Erro", "Por favor, desenhe um dígito antes de processar.")
            return
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)
        cropped_image = image_array[x0:x1 + 1, y0:y1 + 1]
        digit_image = Image.fromarray(cropped_image).resize((20, 20), Image.Resampling.LANCZOS)

        new_image = Image.new('L', (28, 28), 'black')
        new_image.paste(digit_image, (4, 4))  # Center the digit
        image_array = np.array(new_image) / 255.0
        image_input = image_array.flatten().reshape(1, -1)
        output = self.nn.forward(image_input)
        prediction = np.argmax(output, axis=1)[0]
        messagebox.showinfo("Predição", f"Eu acho que é: {prediction}")

    def clear_canvas(self):
        # Limpa todos os desenhos no Canvas do Tkinter
        self.canvas.delete("all")

        # Reinicia a imagem PIL para um fundo preto
        self.image = Image.new("L", (280, 280), "black")
        self.draw_image = ImageDraw.Draw(self.image)

    def run(self):
        self.window.mainloop()
