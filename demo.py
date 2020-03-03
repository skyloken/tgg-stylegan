import datetime
import os
import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

import config
import dnnlib.tflib as tflib
from training import misc

seed = 100
run_id = 0
snapshot = 30000
# run_id = 1
# snapshot = 32000
width = 512
height = 512
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

    def get_label(self):
        return np.eye(2)[self.label_var.get()]

    def generate_figure(self):
        label = self.get_label()
        rnd = np.random.RandomState(None)
        self.latents = rnd.randn(1, self.Gs.input_shape[1])
        self.dlatents = self.Gs.components.mapping.run(self.latents, [label])
        if not self.is_validation.get():
            dlatent_avg = self.Gs.get_var('dlatent_avg')
            self.dlatents = (self.dlatents - dlatent_avg) * np.reshape([self.truncation_psi_var.get()],
                                                                       [-1, 1, 1]) + dlatent_avg
        images = self.Gs.run(self.latents, [label], truncation_psi=self.truncation_psi_var.get(), truncation_cutoff=15,
                             is_validation=self.is_validation.get(), randomize_noise=False, output_transform=fmt)
        self.current_image = Image.fromarray(images[0], 'RGB')
        return ImageTk.PhotoImage(image=self.current_image.resize((512, 512)))

    def generate_transition_animation(self):
        label = self.get_label()
        images = []
        num_split = 50
        rnd = np.random.RandomState(None)
        dst_latents = rnd.randn(1, self.Gs.input_shape[1])

        for i in range(num_split + 1):
            latents = self.latents + (dst_latents - self.latents) * i / num_split
            images_out = self.Gs.run(latents, [label], truncation_psi=self.truncation_psi_var.get(),
                                     truncation_cutoff=15,
                                     is_validation=self.is_validation.get(), randomize_noise=False,
                                     output_transform=fmt)
            self.current_image = Image.fromarray(images_out[0], 'RGB')
            images.append(ImageTk.PhotoImage(image=self.current_image.resize((512, 512))))

        self.latents = dst_latents
        self.dlatents = self.Gs.components.mapping.run(self.latents, [label])
        if not self.is_validation.get():
            dlatent_avg = self.Gs.get_var('dlatent_avg')
            self.dlatents = (self.dlatents - dlatent_avg) * np.reshape([self.truncation_psi_var.get()],
                                                                       [-1, 1, 1]) + dlatent_avg

        return images

    def generate_style_changed_figure(self):

        label = self.get_label()
        rnd = np.random.RandomState(None)

        dst_latents = rnd.randn(1, self.Gs.input_shape[1])
        dst_dlatents = self.Gs.components.mapping.run(dst_latents, [label])

        if self.style_var.get() == 0:
            self.dlatents[:, 4:8] = dst_dlatents[:, 4:8]
        elif self.style_var.get() == 1:
            self.dlatents[:, 8:14] = dst_dlatents[:, 8:14]
        elif self.style_var.get() == 2:
            self.dlatents[:, 0:13] = dst_dlatents[:, 0:13]
        elif self.style_var.get() == 3:
            self.dlatents[:, 13:14] = dst_dlatents[:, 13:14]
        images = self.Gs.components.synthesis.run(self.dlatents, randomize_noise=False, output_transform=fmt)
        self.current_image = Image.fromarray(images[0], 'RGB')
        return ImageTk.PhotoImage(image=self.current_image.resize((512, 512)))

    def create_widgets(self):
        global img

        # values
        self.label_var = tk.IntVar()
        self.label_var.set(0)
        self.is_validation = tk.BooleanVar()
        self.is_validation.set(False)
        self.is_animation = tk.BooleanVar()
        self.is_animation.set(False)
        self.is_loop = tk.BooleanVar()
        self.is_loop.set(False)
        self.truncation_psi_var = tk.DoubleVar()
        self.truncation_psi_var.set(0.7)
        self.style_var = tk.IntVar()
        self.style_var.set(3)

        # Figure canvas
        img = self.generate_figure()
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor='nw', image=img)

        # Label
        self.western_radio_button = tk.Radiobutton(self, value=0, variable=self.label_var, text='Western')
        self.japanese_radio_button = tk.Radiobutton(self, value=1, variable=self.label_var, text='Japanese')

        # Style button
        self.middle_radio_button = tk.Radiobutton(self, value=0, variable=self.style_var, text='Middle')
        self.fine_radio_button = tk.Radiobutton(self, value=1, variable=self.style_var, text='Fine')
        self.pattern_radio_button = tk.Radiobutton(self, value=2, variable=self.style_var, text='Pattern')
        self.color_radio_button = tk.Radiobutton(self, value=3, variable=self.style_var, text='Color')

        # Settings
        self.validation_checkbox = tk.Checkbutton(self, text='Validation', variable=self.is_validation)
        self.animation_checkbox = tk.Checkbutton(self, text='Animation', variable=self.is_animation)
        self.loop_checkbox = tk.Checkbutton(self, text='Loop', variable=self.is_loop)
        self.truncation_psi_scale = tk.Scale(self, orient="horizontal", variable=self.truncation_psi_var, length=400,
                                             from_=-1.0, to=1.0, resolution=0.1)

        # Generate button
        self.generate_button = tk.Button(self, text='Generate', command=self.generate)
        self.change_style_button = tk.Button(self, text='Change style', command=self.change_style)

        # Save button
        self.save_button = tk.Button(self, text='Save image', command=self.save_image)

        # Quit button
        self.quit_button = tk.Button(self, text='Quit', fg='red', command=self.master.destroy)

        # Pack
        self.canvas.pack()
        self.western_radio_button.pack()
        self.japanese_radio_button.pack()
        self.validation_checkbox.pack()
        self.animation_checkbox.pack()
        self.loop_checkbox.pack()
        self.truncation_psi_scale.pack()
        self.generate_button.pack()
        self.middle_radio_button.pack()
        self.fine_radio_button.pack()
        self.pattern_radio_button.pack()
        self.color_radio_button.pack()
        self.change_style_button.pack()
        self.save_button.pack()
        self.quit_button.pack()

    def generate(self):
        global img

        if self.is_animation.get():
            self.generate_button['text'] = 'Generating...'
            self.generate_button['state'] = 'disable'

            while True:
                images = self.generate_transition_animation()
                for image in images:
                    img = image
                    self.canvas.itemconfig(self.image_on_canvas, image=img)
                    self.canvas.update()
                    time.sleep(0.05)
                if not self.is_loop.get():
                    break

            self.generate_button['text'] = 'Generate'
            self.generate_button['state'] = 'normal'
        else:
            img = self.generate_figure()
            self.canvas.itemconfig(self.image_on_canvas, image=img)

    def change_style(self):
        global img

        img = self.generate_style_changed_figure()
        self.canvas.itemconfig(self.image_on_canvas, image=img)

    def save_image(self):
        now = datetime.datetime.now()
        filename = 'gen_' + now.strftime('%H%M%S') + '-%s-t%1.1f' % (
        self.get_label(), self.truncation_psi_var.get()) + '.png'
        self.current_image.save(os.path.join(config.output_dir, filename))


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
