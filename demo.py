import tkinter as tk
import os
import pickle
import numpy as np
from PIL import Image, ImageTk
import dnnlib
import dnnlib.tflib as tflib
import config
from training import misc
from generate_figures import *
import random
import time

seed = 100
run_id = 1
snapshot = None
width = 256
height = 256
global img


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('Takeda Goichi Generator')
        self.pack()
        self.init_model()
        self.create_widgets()

    def init_model(self):
        tflib.init_tf()
        network_pkl = misc.locate_network_pkl(run_id, snapshot)
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, self.Gs = misc.load_pkl(network_pkl)

    def generate_figure(self):
        label = np.identity(2)[self.var.get()]
        rnd = np.random.RandomState(None)
        self.latents = rnd.randn(1, self.Gs.input_shape[1])
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = self.Gs.run(self.latents, [label], truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        return Image.fromarray(images[0], 'RGB')

    def generate_transition_animation(self):

        images = []
        num_split = 39
        rnd = np.random.RandomState(None)
        dst_latents = rnd.randn(1, self.Gs.input_shape[1])
        for i in range(40):
            latents = dst_latents + (self.latents - dst_latents) * i / num_split
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = self.Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
            images.append(ImageTk.PhotoImage(image=PIL.Image.fromarray(images[0], 'RGB')))

        return images

    def create_widgets(self):
        global img
        self.var = tk.IntVar()
        self.var.set(0)

        # Figure canvas
        img = self.generate_figure()
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=img)

        # Label button
        self.radio_button1 = tk.Radiobutton(self, value=0, variable=self.var, text='Western')
        self.radio_button2 = tk.Radiobutton(self, value=1, variable=self.var, text='Japanese')

        # Generate button
        self.generate_button = tk.Button(self, text='Generate', command=self.transition)

        # Quit button
        self.quit_button = tk.Button(self, text='Quit', fg='red', command=self.master.destroy)

        # Pack
        self.canvas.pack()
        self.radio_button1.pack()
        self.radio_button2.pack()
        self.generate_button.pack()
        self.quit_button.pack()

    def generate(self):
        global img
        img = self.generate_figure()
        self.canvas.itemconfig(self.image_on_canvas, image=img)

    def transition(self):
        global img
        self.generate_button['state'] = 'disable'

        images = self.generate_transition_animation()
        for image in images:
            img = image
            self.canvas.itemconfig(self.image_on_canvas, image=img)
            self.canvas.update()
            time.sleep(0.1)

        self.generate_button['state'] = 'normal'


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
