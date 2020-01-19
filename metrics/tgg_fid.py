"""Frechet Inception Distance (FID) for TGG."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib
import config

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

        for fake_label in labels:
            for real_label in labels:
                # Calculate statistics for reals.
                cache_file = self._get_cache_file_for_reals(num_images=self.num_images)
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                if os.path.isfile(cache_file):
                    mu_real, sigma_real = misc.load_pkl(cache_file)
                else:
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
                    misc.save_pkl((mu_real, sigma_real), cache_file)

                # Construct TensorFlow graph.
                result_expr = []
                for gpu_idx in range(num_gpus):
                    with tf.device('/gpu:%d' % gpu_idx):
                        Gs_clone = Gs.clone()
                        inception_clone = inception.clone()

                        latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                        images = Gs_clone.get_output_for(latents, [fake_label] * minibatch_size, is_validation=True,
                                                         randomize_noise=True)

                        images = tflib.convert_images_to_uint8(images)
                        result_expr.append(inception_clone.get_output_for(images))

                # Calculate statistics for fakes.
                for begin in range(0, self.num_images, minibatch_size):
                    end = min(begin + minibatch_size, self.num_images)
                    activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end - begin]
                mu_fake = np.mean(activations, axis=0)
                sigma_fake = np.cov(activations, rowvar=False)

                # Calculate FID.
                m = np.square(mu_fake - mu_real).sum()
                s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)  # pylint: disable=no-member
                dist = m + np.trace(sigma_fake + sigma_real - 2 * s)
                fid = np.real(dist)
                print('Fake label:', fake_label, 'Real label:', real_label, 'FID:', fid)
                self._report_result(fid)

# ----------------------------------------------------------------------------
