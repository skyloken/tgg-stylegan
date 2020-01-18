# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Loss functions."""

import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary


# ----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]


# ----------------------------------------------------------------------------
# WGAN & WGAN-GP loss functions.

def G_wgan(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -fake_scores_out
    return loss


def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels,  # pylint: disable=unused-argument
           wgan_epsilon=0.001):  # Weight for the epsilon term, \epsilon_{drift}.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss


def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels,  # pylint: disable=unused-argument
              wgan_lambda=10.0,  # Weight for the gradient penalty term.
              wgan_epsilon=0.001,  # Weight for the epsilon term, \epsilon_{drift}.
              wgan_target=1.0):  # Target value for gradient magnitudes.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1, 2, 3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target ** 2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss


# ----------------------------------------------------------------------------
# Hinge loss functions. (Use G_wgan with these)

def D_hinge(G, D, opt, training_set, minibatch_size, reals, labels):  # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1. + fake_scores_out) + tf.maximum(0., 1. - real_scores_out)
    return loss


def D_hinge_gp(G, D, opt, training_set, minibatch_size, reals, labels,  # pylint: disable=unused-argument
               wgan_lambda=10.0,  # Weight for the gradient penalty term.
               wgan_target=1.0):  # Target value for gradient magnitudes.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1. + fake_scores_out) + tf.maximum(0., 1. - real_scores_out)

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1, 2, 3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target ** 2))
    return loss


# ----------------------------------------------------------------------------
# Loss functions advocated by the paper
# "Which Training Methods for GANs do actually Converge?"

def G_logistic_saturating(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -tf.nn.softplus(fake_scores_out)  # log(1 - logistic(fake_scores_out))
    return loss


def G_logistic_nonsaturating(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = tf.nn.softplus(-fake_scores_out)  # -log(logistic(fake_scores_out))
    return loss


def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels):  # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out)  # -log(1 - logistic(fake_scores_out))
    loss += tf.nn.softplus(
        -real_scores_out)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type
    return loss


def D_logistic_simplegp(G, D, opt, training_set, minibatch_size, reals, labels, r1_gamma=10.0,
                        r2_gamma=0.0):  # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out)  # -log(1 - logistic(fake_scores_out))
    loss += tf.nn.softplus(
        -real_scores_out)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type

    if r1_gamma != 0.0:
        with tf.name_scope('R1Penalty'):
            real_loss = opt.apply_loss_scaling(tf.reduce_sum(real_scores_out))
            real_grads = opt.undo_loss_scaling(fp32(tf.gradients(real_loss, [reals])[0]))
            r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
            r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        loss += r1_penalty * (r1_gamma * 0.5)

    if r2_gamma != 0.0:
        with tf.name_scope('R2Penalty'):
            fake_loss = opt.apply_loss_scaling(tf.reduce_sum(fake_scores_out))
            fake_grads = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss, [fake_images_out])[0]))
            r2_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1, 2, 3])
            r2_penalty = autosummary('Loss/r2_penalty', r2_penalty)
        loss += r2_penalty * (r2_gamma * 0.5)
    return loss


# ----------------------------------------------------------------------------
# Loss functions for Takeda Goichi Generator


def mix_dlatents(src_dlatents, dst_dlatents):
    def mix_middle(x):
        return tf.concat([tf.gather(x[1], list(range(0, 4))), tf.gather(x[0], list(range(4, 8))),
                          tf.gather(x[1], list(range(8, 14)))], 0)

    def mix_fine(x):
        return tf.concat([tf.gather(x[1], list(range(0, 8))), tf.gather(x[0], list(range(8, 14)))], 0)

    stacked_dlatents = tf.stack([src_dlatents, dst_dlatents], axis=1)
    middle_dlatents = tf.map_fn(mix_middle, stacked_dlatents)
    fine_dlatents = tf.map_fn(mix_fine, stacked_dlatents)
    return tf.concat([middle_dlatents, fine_dlatents], 0)


def G_tgg(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument

    # Basic Generator Loss
    half_minibatch_size = minibatch_size // 2
    wst_latents = tf.random_normal([half_minibatch_size] + G.input_shapes[0][1:])
    wst_labels = tf.one_hot(tf.fill([half_minibatch_size], 0), 2, dtype=tf.int32)
    jpn_latents = tf.random_normal([half_minibatch_size] + G.input_shapes[0][1:])
    jpn_labels = tf.one_hot(tf.fill([half_minibatch_size], 1), 2, dtype=tf.int32)
    fake_images_out = G.get_output_for(tf.concat([wst_latents, jpn_latents], 0), tf.concat([wst_labels, jpn_labels], 0),
                                       is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, tf.concat([wst_labels, jpn_labels], 0), is_training=True))

    # Western and Japanese Mixing Loss
    wst_dlatents = G.components.mapping.get_output_for(wst_latents, wst_labels, is_training=True)
    jpn_dlatents = G.components.mapping.get_output_for(jpn_latents, jpn_labels, is_training=True)
    mixed_dlatents = mix_dlatents(wst_dlatents, jpn_dlatents)
    mixed_images_out = G.components.synthesis.get_output_for(mixed_dlatents, is_training=True)
    mixed_wst_scores_out = fp32(
        D.get_output_for(mixed_images_out, tf.concat([wst_labels, wst_labels], 0), is_training=True))
    mixed_jpn_scores_out = fp32(
        D.get_output_for(mixed_images_out, tf.concat([jpn_labels, jpn_labels], 0), is_training=True))

    # Loss
    loss = tf.nn.softplus(-fake_scores_out)  # -log(logistic(fake_scores_out))
    loss += tf.nn.softplus(-mixed_wst_scores_out)
    loss += tf.nn.softplus(-mixed_jpn_scores_out)
    return loss


def G_tgg2(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument

    # Basic Generator Loss
    half_minibatch_size = minibatch_size // 2
    wst_latents = tf.random_normal([half_minibatch_size] + G.input_shapes[0][1:])
    wst_labels = tf.one_hot(tf.fill([half_minibatch_size], 0), 2, dtype=tf.int32)
    jpn_latents = tf.random_normal([half_minibatch_size] + G.input_shapes[0][1:])
    jpn_labels = tf.one_hot(tf.fill([half_minibatch_size], 1), 2, dtype=tf.int32)
    fake_images_out = G.get_output_for(tf.concat([wst_latents, jpn_latents], 0), tf.concat([wst_labels, jpn_labels], 0),
                                       is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, tf.concat([wst_labels, jpn_labels], 0), is_training=True))

    # Western and Japanese Mixing Loss
    wst_dlatents = G.components.mapping.get_output_for(wst_latents, wst_labels, is_training=True)
    jpn_dlatents = G.components.mapping.get_output_for(jpn_latents, jpn_labels, is_training=True)
    mixed_dlatents = mix_dlatents(wst_dlatents, jpn_dlatents)
    mixed_images_out = G.components.synthesis.get_output_for(mixed_dlatents, is_training=True)
    mixed_scores_out = fp32(
        D.get_output_for(mixed_images_out, tf.ones([minibatch_size, 2], dtype=tf.int32), is_training=True))

    # Loss
    loss = tf.nn.softplus(-fake_scores_out)  # -log(logistic(fake_scores_out))
    loss += tf.nn.softplus(-mixed_scores_out)
    return loss


def G_tgg3(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])

    # Basic Generator Loss
    wj_minibatch_size = minibatch_size // 3
    labels = training_set.get_random_labels_tf(minibatch_size - wj_minibatch_size)
    wj_labels = tf.ones([wj_minibatch_size, 2], dtype=tf.float32)

    fake_images_out = G.get_output_for(latents, tf.concat([labels, wj_labels], 0), is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, tf.concat([labels, wj_labels], 0), is_training=True))

    loss = tf.nn.softplus(-fake_scores_out)  # -log(logistic(fake_scores_out))

    return loss
