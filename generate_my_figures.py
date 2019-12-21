import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from training import misc
from generate_figures import *

run_id = 31
snapshot = None
width = 256
height = 256

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

def generate_figures(Gs):

    # Generate western images
    for i in range(250):
        # Pick latent vector.
        # latents = np.random.RandomState(i).randn(1, Gs.input_shape[1])
        latents = np.stack([np.random.RandomState(i).randn(Gs.input_shape[1])])
        # dlatents = Gs.components.mapping.run(latents, [[1, 0]])
        dlatents = Gs.components.mapping.run(latents, [[1, 0, 0, 0, 0]])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        # images = Gs.run(latents, [[1, 0]], randomize_noise=False, output_transform=fmt)
        images = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)

        # Save image.
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'gen_western_images'), exist_ok=True)
        png_filename = os.path.join(config.output_dir, 'gen_western_images', 'gen_%s.png' % i)
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
    
    # Generate japanese images
    for i in range(250):
        # Pick latent vector.
        # latents = np.random.RandomState(i).randn(1, Gs.input_shape[1])
        latents = np.stack([np.random.RandomState(i).randn(Gs.input_shape[1])])
        # dlatents = Gs.components.mapping.run(latents, [[0, 1]])
        dlatents = Gs.components.mapping.run(latents, [[0, 1, 0, 0, 0]])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        # images = Gs.run(latents, [[0, 1]], randomize_noise=False, output_transform=fmt)
        images = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)

        # Save image.
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'gen_japanese_images'), exist_ok=True)
        png_filename = os.path.join(config.output_dir, 'gen_japanese_images', 'gen_%s.png' % i)
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

def draw_color_conditions(Gs, src_seed, label):
    src_latents = np.random.RandomState(src_seed).randn(1, Gs.input_shape[1])
    src_dlatents = Gs.components.mapping.run(src_latents, [label])
    src_image = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)[0]
    png_filename = os.path.join(config.output_dir, 'src.png')
    PIL.Image.fromarray(src_image, 'RGB').save(png_filename)

    color_dlatents = Gs.components.mapping.run(np.random.RandomState(src_seed).randn(3, Gs.input_shape[1]), [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    color_images = Gs.components.synthesis.run(color_dlatents, randomize_noise=False, **synthesis_kwargs)
    PIL.Image.fromarray(color_images[0], 'RGB').save(os.path.join(config.output_dir, 'red.png'))
    PIL.Image.fromarray(color_images[1], 'RGB').save(os.path.join(config.output_dir, 'blue.png'))
    PIL.Image.fromarray(color_images[2], 'RGB').save(os.path.join(config.output_dir, 'green.png'))

    colored_dlatents = np.stack([src_dlatents[0]] * len(color_dlatents))
    colored_dlatents[:, 13] = color_dlatents[:, 13]
    images = Gs.components.synthesis.run(colored_dlatents, randomize_noise=False, **synthesis_kwargs)
    
    for i, image in enumerate(images):
        png_filename = os.path.join(config.output_dir, 'gen_%s.png' % i)
        PIL.Image.fromarray(image, 'RGB').save(png_filename)

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load network.
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    print('Loading networks from "%s"...' % network_pkl)
    G, D, Gs = misc.load_pkl(network_pkl)
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Generate figures.
    os.makedirs(config.output_dir, exist_ok=True)
    # generate_figures(Gs)

    latents = np.random.RandomState(100).randn(1, Gs.input_shape[1])
    dlatents = Gs.components.mapping.run(latents, [[1, 0]])
    print(dlatents)

    # draw_color_conditions(Gs, src_seed=200, label=[1, 0, 0, 0, 0])

    # draw_western_and_japanese_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-coarse.png'), Gs, w=width, h=height, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(0,4)]*6)
    # draw_western_and_japanese_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-middle.png'), Gs, w=width, h=height, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(4,8)]*6)
    # draw_western_and_japanese_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-fine.png'), Gs, w=width, h=height, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(8,14)]*6)
    
    # draw_western_and_japanese_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-coarse.png'), Gs, w=width, h=height, src_seeds=[81,175,186,206], dst_seeds=[7,40,61,100], style_ranges=[range(0,4)]*4)
    # draw_western_and_japanese_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-middle.png'), Gs, w=width, h=height, src_seeds=[81,175,186,206], dst_seeds=[7,40,61,100], style_ranges=[range(4,8)]*4)
    # draw_western_and_japanese_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-fine.png'), Gs, w=width, h=height, src_seeds=[81,175,186,206], dst_seeds=[7,40,61,100], style_ranges=[range(8,14)]*4)
    # draw_japanese_and_western_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-coarse-2.png'), Gs, w=width, h=height, dst_seeds=[81,175,186,206], src_seeds=[7,40,61,100], style_ranges=[range(0,4)]*4)
    # draw_japanese_and_western_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-middle-2.png'), Gs, w=width, h=height, dst_seeds=[81,175,186,206], src_seeds=[7,40,61,100], style_ranges=[range(4,8)]*4)
    # draw_japanese_and_western_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-fine-2.png'), Gs, w=width, h=height, dst_seeds=[81,175,186,206], src_seeds=[7,40,61,100], style_ranges=[range(8,14)]*4)

    # draw_style_detail(os.path.join(config.output_dir, 'style-detail-1.png'), Gs, w=width, h=height, label=[1, 0, 0, 0, 0], src_seed=223, dst_seeds=[5, 9, 13, 36, 111, 113, 245, 248], style_range=1)
    # draw_style_detail(os.path.join(config.output_dir, 'style-detail-2.png'), Gs, w=width, h=height, label=[1, 0, 0, 0, 0], src_seed=223, dst_seeds=[5, 9, 13, 36, 111, 113, 245, 248], style_range=2)
    # draw_style_detail(os.path.join(config.output_dir, 'style-detail-3.png'), Gs, w=width, h=height, label=[1, 0, 0, 0, 0], src_seed=223, dst_seeds=[5, 9, 13, 36, 111, 113, 245, 248], style_range=3)
    # draw_style_detail(os.path.join(config.output_dir, 'style-detail-4.png'), Gs, w=width, h=height, label=[1, 0, 0, 0, 0], src_seed=223, dst_seeds=[5, 9, 13, 36, 111, 113, 245, 248], style_range=4)
    # draw_style_detail(os.path.join(config.output_dir, 'style-detail-5.png'), Gs, w=width, h=height, label=[1, 0, 0, 0, 0], src_seed=223, dst_seeds=[5, 9, 13, 36, 111, 113, 245, 248], style_range=5)

    # draw_uncurated_result_figure(os.path.join(config.output_dir, 'uncurated.png'), Gs, cx=0, cy=0, cw=width, ch=height, rows=3, lods=[0,1,2,2,3,3], seed=5)
    # draw_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-coarse.png'), Gs, w=width, h=height, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(0,4)]*6)
    # draw_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-middle.png'), Gs, w=width, h=height, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(4,8)]*6)
    # draw_style_mixing_figure(os.path.join(config.output_dir, 'style-mixing-fine.png'), Gs, w=width, h=height, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(8,14)]*6)
    # draw_noise_detail_figure(os.path.join(config.output_dir, 'noise-detail.png'), Gs, w=width, h=height, num_samples=100, seeds=[1157,1012])
    # draw_noise_components_figure(os.path.join(config.output_dir, 'noise-components.png'), Gs, w=width, h=height, seeds=[1967,1555], noise_ranges=[range(0, 18), range(0, 0), range(8, 18), range(0, 8)], flips=[1])
    # draw_truncation_trick_figure(os.path.join(config.output_dir, 'truncation-trick.png'), Gs, w=width, h=height, seeds=[91,388], psis=[1, 0.7, 0.5, 0, -0.5, -1])
    # draw_transition_figure(os.path.join(config.output_dir, 'transition_%s_to_%s.gif' % (9, 16)), Gs, w=width, h=height, src_seed=9, dst_seed=16)

if __name__ == "__main__":
    main()