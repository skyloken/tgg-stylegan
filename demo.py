import tkinter as tk
import os
import pickle
import numpy as np
from PIL import Image, ImageTk
import dnnlib
import dnnlib.tflib as tflib
import config
from training import misc
import random
import time

seed = 100
run_id = 0
snapshot = None
width = 256
height = 256
fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
global img


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('Demo')
        self.pack()
        self.init_model()
        self.create_widgets()

    def init_model(self):
        tflib.init_tf()
        network_pkl = misc.locate_network_pkl(run_id, snapshot)
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, self.Gs = misc.load_pkl(network_pkl)

    def generate_figure(self):
        label = np.identity(2)[self.label_var.get()]
        rnd = np.random.RandomState(None)
        self.latents = rnd.randn(1, self.Gs.input_shape[1])
        self.dlatents = self.Gs.components.mapping.run(self.latents, [label])
        images = self.Gs.run(self.latents, [label], truncation_psi=self.truncation_psi_var.get(), randomize_noise=True,
                             output_transform=fmt)
        return ImageTk.PhotoImage(image=Image.fromarray(images[0], 'RGB'))

    def generate_transition_animation(self):
        label = np.identity(2)[self.label_var.get()]
        images = []
        num_split = 100
        rnd = np.random.RandomState(None)
        dst_latents = rnd.randn(1, self.Gs.input_shape[1])

        for i in range(num_split + 1):
            latents = self.latents + (dst_latents - self.latents) * i / num_split
            images_out = self.Gs.run(latents, [label], truncation_psi=self.truncation_psi_var.get(),
                                     randomize_noise=False, output_transform=fmt)
            images.append(ImageTk.PhotoImage(image=Image.fromarray(images_out[0], 'RGB')))

        self.latents = dst_latents
        self.dlatents = self.Gs.components.mapping.run(self.latents, [label])

        return images

    def generate_color_changed_figure(self):

        label = np.identity(2)[self.label_var.get()]
        rnd = np.random.RandomState(None)
        dst_latents = rnd.randn(1, self.Gs.input_shape[1])
        dst_dlatents = self.Gs.components.mapping.run(dst_latents, [label])

        self.dlatents[:, 13] = dst_dlatents[:, 13]
        images = self.Gs.components.synthesis.run(self.dlatents, truncation_psi=self.truncation_psi_var.get(),
                                                  randomize_noise=False, output_transform=fmt)
        return ImageTk.PhotoImage(image=Image.fromarray(images[0], 'RGB'))

    def create_widgets(self):
        global img

        # values
        self.label_var = tk.IntVar()
        self.label_var.set(0)
        self.is_animation = tk.BooleanVar()
        self.is_animation.set(False)
        self.truncation_psi_var = tk.IntVar()
        self.truncation_psi_var.set(0.7)

        # Figure canvas
        img = self.generate_figure()
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=img)

        # Label button
        self.western_radio_button = tk.Radiobutton(self, value=0, variable=self.label_var, text='Western')
        self.japanese_radio_button = tk.Radiobutton(self, value=1, variable=self.label_var, text='Japanese')

        # Settings
        self.animation_checkbox = tk.Checkbutton(text='Animation', variable=self.is_animation)
        self.truncation_psi_scale = tk.Scale(self, orient="horizontal", variable=self.truncation_psi_var)

        # Generate button
        self.generate_button = tk.Button(self, text='Generate', command=self.generate)
        self.change_color_button = tk.Button(self, text='Change color', command=self.change_color)

        # Quit button
        self.quit_button = tk.Button(self, text='Quit', fg='red', command=self.master.destroy)

        # Pack
        self.canvas.pack()
        self.western_radio_button.pack()
        self.japanese_radio_button.pack()
        self.animation_checkbox.pack()
        self.truncation_psi_scale.pack()
        self.generate_button.pack()
        self.change_color_button.pack()
        self.quit_button.pack()

    def generate(self):
        global img

        if self.is_animation.get():
            self.generate_button['text'] = 'Generating...'
            self.generate_button['state'] = 'disable'

            images = self.generate_transition_animation()
            for image in images:
                img = image
                self.canvas.itemconfig(self.image_on_canvas, image=img)
                self.canvas.update()
                time.sleep(0.05)

            self.generate_button['text'] = 'Transition'
            self.generate_button['state'] = 'normal'
        else:
            img = self.generate_figure()
            self.canvas.itemconfig(self.image_on_canvas, image=img)

    def change_color(self):
        global img

        img = self.generate_color_changed_figure()
        self.canvas.itemconfig(self.image_on_canvas, image=img)


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
