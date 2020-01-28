import os

import PIL.Image
import numpy as np

import config
import dnnlib.tflib as tflib
from training import misc

run_id = 0
snapshot = None
width = 256
height = 256

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
synthesis_kwargs = dict(output_transform=fmt, minibatch_size=8)


# ----------------------------------------------------------------------------
# Generate figures

def generate_figures(Gs, gen_num=100, truncation_psi=0.7, truncation_cutoff=8, randomize_noise=True):
    # Generate western images
    for i in range(gen_num):
        # Pick latent vector.
        latents = np.random.RandomState(i).randn(1, Gs.input_shape[1])

        # Generate image.
        wst_images = Gs.run(latents, [[1, 0]], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                            randomize_noise=randomize_noise, output_transform=fmt)
        jpn_images = Gs.run(latents, [[0, 1]], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                            randomize_noise=randomize_noise, output_transform=fmt)

        # Save image.
        os.makedirs(config.output_dir, exist_ok=True)

        os.makedirs(os.path.join(config.output_dir, 'gen_western_images'), exist_ok=True)
        PIL.Image.fromarray(wst_images[0], 'RGB').save(
            os.path.join(config.output_dir, 'gen_western_images', 'gen_%s.png' % i))

        os.makedirs(os.path.join(config.output_dir, 'gen_japanese_images'), exist_ok=True)
        PIL.Image.fromarray(jpn_images[0], 'RGB').save(
            os.path.join(config.output_dir, 'gen_japanese_images', 'gen_%s.png' % i))

    # Generate japanese images
    for i in range(gen_num):
        # Pick latent vector.
        latents = np.random.RandomState(i).randn(1, Gs.input_shape[1])

        # Generate image.
        images = Gs.run(latents, [[0, 1]], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                        randomize_noise=randomize_noise, output_transform=fmt)

        # Save image.
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'gen_japanese_images'), exist_ok=True)
        png_filename = os.path.join(config.output_dir, 'gen_japanese_images', 'gen_%s.png' % i)
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


def generate_mixings(Gs, style_range, src_label=[1, 0], gen_num=100, truncation_psi=0.7, randomize_noise=True):
    dlatent_avg = Gs.get_var('dlatent_avg')

    for i in range(gen_num):
        # Pick latent vector.
        rand = np.random.RandomState(i)
        wst_latents = rand.randn(1, Gs.input_shape[1])
        jpn_latents = rand.randn(1, Gs.input_shape[1])
        wst_dlatents = Gs.components.mapping.run(wst_latents, [[1, 0]])
        jpn_dlatents = Gs.components.mapping.run(jpn_latents, [[0, 1]])

        # Truncation trick
        wst_dlatents = (wst_dlatents - dlatent_avg) * np.reshape([truncation_psi], [-1, 1, 1]) + dlatent_avg
        jpn_dlatents = (jpn_dlatents - dlatent_avg) * np.reshape([truncation_psi], [-1, 1, 1]) + dlatent_avg

        # Style Mixing
        if src_label == [1, 0]:
            dlatents = np.copy(jpn_dlatents)
            dlatents[:, style_range] = wst_dlatents[:, style_range]
        elif src_label == [0, 1]:
            dlatents = np.copy(wst_dlatents)
            dlatents[:, style_range] = jpn_dlatents[:, style_range]
        images = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)

        # Save
        dir_name = 'style_mixing-{0}-{1}-psi{2}'.format(src_label, style_range, truncation_psi)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, dir_name), exist_ok=True)
        PIL.Image.fromarray(images[0], 'RGB').save(
            os.path.join(config.output_dir, dir_name, 'gen_%s.png' % i))


# ----------------------------------------------------------------------------
# Western and Japanese style mixing.

def draw_western_and_japanese_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    print(png)
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
    src_dlatents = Gs.components.mapping.run(src_latents, [[1, 0]] * len(src_seeds))  # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, [[0, 1]] * len(dst_seeds))  # [seed, layer, component]
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)


def draw_japanese_and_western_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    print(png)
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
    src_dlatents = Gs.components.mapping.run(src_latents, [[0, 1]] * len(src_seeds))  # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, [[1, 0]] * len(dst_seeds))  # [seed, layer, component]
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)


# ----------------------------------------------------------------------------
# Style detail

def draw_style_detail(png, Gs, w, h, label, src_seed, dst_seeds, style_range):
    print(png)

    src_latents = np.random.RandomState(src_seed).randn(1, Gs.input_shape[1])
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)

    src_dlatents = Gs.components.mapping.run(src_latents, [label])
    dst_dlatents = Gs.components.mapping.run(dst_latents, [[0, 1, 0, 0, 0]] * len(dst_seeds))
    src_image = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)[0]
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    style_len = src_dlatents.shape[1]
    canvas = PIL.Image.new('RGB', (w * ((style_len - style_range + 1) + 1), h * (len(dst_seeds) + 1)), 'white')
    canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), (0, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))

        row_dlatents = np.stack([src_dlatents[0]] * (style_len - style_range + 1))
        for i in range(0, len(row_dlatents)):
            row_dlatents[i, i:i + style_range] = dst_dlatents[row, i:i + style_range]

        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)


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
    # generate_figures(Gs, gen_num=100, truncation_psi=0.7, truncation_cutoff=8, randomize_noise=True)
    # generate_mixings(Gs, style_range=range(0, 10), src_label=[1, 0], gen_num=500, truncation_psi=1.0, randomize_noise=True)
    generate_mixings(Gs, style_range=range(0, 7), src_label=[0, 1], gen_num=500, truncation_psi=0.7,
                     randomize_noise=True)

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
