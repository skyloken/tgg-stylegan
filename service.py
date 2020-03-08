import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image

import config
import dnnlib.tflib as tflib
from training import misc

run_id = 0
snapshot = None
width = 256
height = 256

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
synthesis_kwargs = dict(output_transform=fmt, minibatch_size=8)


class TakedaGoichiGenerator:

    def __init__(self):
        self.init_model()

    def init_model(self):
        self.session = tflib.init_tf()
        network_pkl = misc.locate_network_pkl(run_id, snapshot)
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, self.Gs = misc.load_pkl(network_pkl)
        self.Gs.print_layers()

    def generate_figure(self, label):
        latents = np.random.randn(1, self.Gs.input_shape[1])
        with self.session.as_default():
            images = self.Gs.run(latents, [label], is_validation=True, randomize_noise=False, output_transform=fmt)
        return latents[0], Image.fromarray(images[0], 'RGB')

    def mix_styles(self, wst_latent, jpn_latent):

        style_ranges = [range(0, i) for i in range(14, -1, -1)]

        mixed_images = []

        labels = np.eye(2).tolist()
        with self.session.as_default():
            for src_label in labels:
                for style_range in style_ranges:
                    # Pick latent vector.
                    wst_dlatents = self.Gs.components.mapping.run(np.array([wst_latent]), [[1, 0]])
                    jpn_dlatents = self.Gs.components.mapping.run(np.array([jpn_latent]), [[0, 1]])

                    # Style Mixing
                    if src_label == [1, 0]:
                        dlatents = np.copy(jpn_dlatents)
                        dlatents[:, style_range] = wst_dlatents[:, style_range]
                    elif src_label == [0, 1]:
                        dlatents = np.copy(wst_dlatents)
                        dlatents[:, style_range] = jpn_dlatents[:, style_range]
                    
                    images = self.Gs.components.synthesis.run(
                        dlatents, randomize_noise=False, **synthesis_kwargs)
                    
                    mixed_images.append(Image.fromarray(images[0], 'RGB'))

        return mixed_images
