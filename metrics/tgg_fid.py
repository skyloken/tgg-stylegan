"""Frechet Inception Distance (FID) for TGG."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib
import config
import pandas as pd

from metrics import metric_base
from training import misc


# ----------------------------------------------------------------------------

class TGG_FID(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu
    
    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        # inception = misc.load_pkl('https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn')  # inception_v3_features.pkl
        inception = misc.load_pkl(os.path.join(config.data_dir, '..', 'data', 'inception_v3_features.pkl'))
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        label_size = 2
        labels = np.eye(label_size).tolist()

        style_patterns = [(range(0, i), range(i, 14)) for i in range(14, -1, -1)]

        # Calculate statistics for reals.
        mu_reals = []
        sigma_reals = []
        for real_label in labels:
            for idx, images in enumerate(
                    self._iterate_reals_for_label(minibatch_size=minibatch_size, label=real_label)):
                begin = idx * minibatch_size
                end = min(begin + minibatch_size, self.num_images)
                activations[begin:end] = inception.run(images[:end - begin], num_gpus=num_gpus,
                                                        assume_frozen=True)
                if end == self.num_images:
                    break
            mu_real = np.mean(activations, axis=0)
            sigma_real = np.cov(activations, rowvar=False)
            mu_reals.append(mu_real)
            sigma_reals.append(sigma_real)

        result_outputs = []
        # TODO: Please change
        src_label = labels[0]
        # src_label = labels[1]
        df_fid = pd.DataFrame()
        for style_pattern in style_patterns:
            cols = []
            for real_label in labels:

                # Construct TensorFlow graph.
                result_expr = []
                for gpu_idx in range(num_gpus):
                    with tf.device('/gpu:%d' % gpu_idx):
                        Gs_clone = Gs.clone()
                        inception_clone = inception.clone()

                        # Style Mixing
                        wst_latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                        wst_labels = tf.one_hot(tf.fill([self.minibatch_per_gpu], 0), 2, dtype=tf.float32)
                        jpn_latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                        jpn_labels = tf.one_hot(tf.fill([self.minibatch_per_gpu], 1), 2, dtype=tf.float32)
                        wst_dlatents = Gs_clone.components.mapping.get_output_for(wst_latents, wst_labels)
                        jpn_dlatents = Gs_clone.components.mapping.get_output_for(jpn_latents, jpn_labels)
                        if src_label == [1, 0]:
                            mixed_dlatents = self._mix_dlatents(wst_dlatents, jpn_dlatents, style_pattern[0], style_pattern[1])
                        elif src_label == [0, 1]:
                            mixed_dlatents = self._mix_dlatents(jpn_dlatents, wst_dlatents, style_pattern[0], style_pattern[1])
                        images = Gs_clone.components.synthesis.get_output_for(mixed_dlatents, randomize_noise=True)

                        images = tflib.convert_images_to_uint8(images)
                        result_expr.append(inception_clone.get_output_for(images))

                # Calculate statistics for fakes.
                for begin in range(0, self.num_images, minibatch_size):
                    end = min(begin + minibatch_size, self.num_images)
                    activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end - begin]
                mu_fake = np.mean(activations, axis=0)
                sigma_fake = np.cov(activations, rowvar=False)

                # Calculate FID.
                if real_label == [1, 0]:
                    mu_real = mu_reals[0]
                    sigma_real = sigma_reals[0]
                elif real_label == [0, 1]:
                    mu_real = mu_reals[1]
                    sigma_real = sigma_reals[1]
                m = np.square(mu_fake - mu_real).sum()
                s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)  # pylint: disable=no-member
                dist = m + np.trace(sigma_fake + sigma_real - 2 * s)
                fid = np.real(dist)
                self._report_result(fid)

                # Output
                result_output = 'Source label:{0}, Style pattern:{1}, Real label:{2}, FID:{3}'.format(src_label, style_pattern, real_label, fid)
                print(result_output)

                # csv
                cols.append(fid)
            df_fid[str(style_pattern)] = cols
        df_fid.to_csv(os.path.join(config.result_dir, 'fid-%s.csv' % src_label))

    def _mix_dlatents(self, src_dlatents, dst_dlatents, src_range, dst_range):
        def mix(x):
            return tf.concat([tf.gather(x[0], list(src_range)), tf.gather(x[1], list(dst_range))], 0)

        if not list(dst_range):
            return src_dlatents
        elif not list(src_range):
            return dst_dlatents
        
        stacked_dlatents = tf.stack([src_dlatents, dst_dlatents], axis=1)
        return tf.map_fn(mix, stacked_dlatents)
# ----------------------------------------------------------------------------
