# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Tool for creating multi-resolution TFRecords datasets for StyleGAN and ProGAN."""

# pylint: disable=too-many-lines
import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue # pylint: disable=import-error
import traceback
import numpy as np
import tensorflow as tf
import PIL.Image
import dnnlib.tflib as tflib
import pickle
from training import dataset

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    exit(1)

#----------------------------------------------------------------------------

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        print("Labels shape: ", labels.shape)
        print(self.cur_images)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

#----------------------------------------------------------------------------

class ExceptionInfo(object):
    def __init__(self):
        self.value = sys.exc_info()[1]
        self.traceback = traceback.format_exc()

#----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

#----------------------------------------------------------------------------

class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = Queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for _idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, '__call__') # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = Queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func): # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            print('\n\nWorker thread caught an exception:\n' + result.traceback)
            raise result.value
        return result, args

    def finish(self):
        for _idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self): # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(self, item_iterator, process_func=lambda x: x, pre_func=lambda x: x, post_func=lambda x: x, max_items_in_flight=None):
        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, _idx):
            return process_func(prepared)

        def retire_result():
            processed, (_prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1

        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result(): yield res
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res

#----------------------------------------------------------------------------

def display(tfrecord_dir):
    print('Loading dataset "%s"' % tfrecord_dir)
    tflib.init_tf({'gpu_options.allow_growth': True})
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
    tflib.init_uninitialized_vars()
    import cv2  # pip install opencv-python

    idx = 0
    while True:
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if idx == 0:
            print('Displaying images')
            cv2.namedWindow('dataset_tool')
            print('Press SPACE or ENTER to advance, ESC to exit')
        print('\nidx = %-8d\nlabel = %s' % (idx, labels[0].tolist()))
        cv2.imshow('dataset_tool', images[0].transpose(1, 2, 0)[:, :, ::-1]) # CHW => HWC, RGB => BGR
        idx += 1
        if cv2.waitKey() == 27:
            break
    print('\nDisplayed %d images.' % idx)

#----------------------------------------------------------------------------

def extract(tfrecord_dir, output_dir):
    print('Loading dataset "%s"' % tfrecord_dir)
    tflib.init_tf({'gpu_options.allow_growth': True})
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size=0, repeat=False, shuffle_mb=0)
    tflib.init_uninitialized_vars()

    print('Extracting images to "%s"' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    idx = 0
    while True:
        if idx % 10 == 0:
            print('%d\r' % idx, end='', flush=True)
        try:
            images, _labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if images.shape[1] == 1:
            img = PIL.Image.fromarray(images[0][0], 'L')
        else:
            img = PIL.Image.fromarray(images[0].transpose(1, 2, 0), 'RGB')
        idx += 1
    print('Extracted %d images.' % idx)

#----------------------------------------------------------------------------

def compare(tfrecord_dir_a, tfrecord_dir_b, ignore_labels):
    max_label_size = 0 if ignore_labels else 'full'
    print('Loading dataset "%s"' % tfrecord_dir_a)
    tflib.init_tf({'gpu_options.allow_growth': True})
    dset_a = dataset.TFRecordDataset(tfrecord_dir_a, max_label_size=max_label_size, repeat=False, shuffle_mb=0)
    print('Loading dataset "%s"' % tfrecord_dir_b)
    dset_b = dataset.TFRecordDataset(tfrecord_dir_b, max_label_size=max_label_size, repeat=False, shuffle_mb=0)
    tflib.init_uninitialized_vars()

    print('Comparing datasets')
    idx = 0
    identical_images = 0
    identical_labels = 0
    while True:
        if idx % 100 == 0:
            print('%d\r' % idx, end='', flush=True)
        try:
            images_a, labels_a = dset_a.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_a, labels_a = None, None
        try:
            images_b, labels_b = dset_b.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_b, labels_b = None, None
        if images_a is None or images_b is None:
            if images_a is not None or images_b is not None:
                print('Datasets contain different number of images')
            break
        if images_a.shape == images_b.shape and np.all(images_a == images_b):
            identical_images += 1
        else:
            print('Image %d is different' % idx)
        if labels_a.shape == labels_b.shape and np.all(labels_a == labels_b):
            identical_labels += 1
        else:
            print('Label %d is different' % idx)
        idx += 1
    print('Identical images: %d / %d' % (identical_images, idx))
    if not ignore_labels:
        print('Identical labels: %d / %d' % (identical_labels, idx))

#----------------------------------------------------------------------------

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_from_images(tfrecord_dir, image_dir, shuffle, add_condition, label_file):
    print("ADD CONDITION ", add_condition)
    print('Loading images from "%s"' % image_dir)

    all_data = unpickle(label_file)
    image_filenames_temp = all_data["Filenames"]
    conditions_all = all_data["Labels"] #for others use Clusters
    assert len(conditions_all) == len(image_filenames_temp)

    import pandas as pd
    df = pd.DataFrame.from_dict(all_data)
    #image_filenames_temp = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    print("NUMBER OF FILES: ", len(image_filenames_temp))
    if len(image_filenames_temp) == 0:
        error('No input images found')

    img = np.asarray(PIL.Image.open(image_dir + df['Filenames'][0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')


    drop = []
    df_copy = df.copy()
    for i in range(len(df["Filenames"])):
        img = np.asarray(PIL.Image.open(image_dir + df["Filenames"].iloc[i]))
        if channels == 1:
            img = img[np.newaxis, :, :] # HW => CHW
        else:
            try:
                img = img.transpose([2, 0, 1]) # HWC => CHW

                print("added")
            except:
                drop.append(i)
                print("deleted")
                continue

    print("NUMBER OF IMAGES BEFORE: ", len(df))
    df = df.drop(drop)
    print("NUMBER OF IMAGES AFTER: ", len(df))

    with TFRecordExporter(tfrecord_dir, len(df)) as tfr:
        order = np.arange(len(df))
        drop = []
        deleted = []
        for idx in range(order.size):
            print("HERE ",df["Filenames"].iloc[order[idx]])
            img = np.asarray(PIL.Image.open(image_dir + df["Filenames"].iloc[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :] # HW => CHW
            else:
                img = img.transpose([2, 0, 1]) # HWC => CHW
                try:
                    tfr.add_image(img)
                except:
                    #os.remove(image_filenames[order[idx]])
                    print("DELETED")
                    drop.append(df.index[order[idx]])
                    print("###########")
                    deleted.append(df["Filenames"].iloc[order[idx]])
                    continue


        print("############# DELETED FILENAMES ############")
        print(deleted)
        df = df.drop(drop)
        if add_condition == 1:
            print("Adding Labels")
            conditions = np.asarray(df["Labels"])
            #labels = np.random.randint(0,np.max(conditions),len(image_filenames))
            onehot = np.zeros((conditions.size, np.max(conditions) + 1), dtype=np.float32)
            onehot[np.arange(conditions.size), conditions] = 1.0
            print(onehot)
            tfr.add_labels(onehot)


#----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating multi-resolution TFRecords datasets for StyleGAN and ProGAN.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'display',          'Display images in dataset.',
                                            'display datasets/mnist')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')

    p = add_command(    'extract',          'Extract images from dataset.',
                                            'extract datasets/mnist mnist-images')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')
    p.add_argument(     'output_dir',       help='Directory to extract the images into')

    p = add_command(    'compare',          'Compare two datasets.',
                                            'compare datasets/mydataset datasets/mnist')
    p.add_argument(     'tfrecord_dir_a',   help='Directory containing first dataset')
    p.add_argument(     'tfrecord_dir_b',   help='Directory containing second dataset')
    p.add_argument(     '--ignore_labels',  help='Ignore labels (default: 0)', type=int, default=0)

    p = add_command(    'create_mnist',     'Create dataset for MNIST.',
                                            'create_mnist datasets/mnist ~/downloads/mnist')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'mnist_dir',        help='Directory containing MNIST')

    p = add_command(    'create_mnistrgb',  'Create dataset for MNIST-RGB.',
                                            'create_mnistrgb datasets/mnistrgb ~/downloads/mnist')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'mnist_dir',        help='Directory containing MNIST')
    p.add_argument(     '--num_images',     help='Number of composite images to create (default: 1000000)', type=int, default=1000000)
    p.add_argument(     '--random_seed',    help='Random seed (default: 123)', type=int, default=123)

    p = add_command(    'create_cifar10',   'Create dataset for CIFAR-10.',
                                            'create_cifar10 datasets/cifar10 ~/downloads/cifar10')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'cifar10_dir',      help='Directory containing CIFAR-10')

    p = add_command(    'create_cifar100',  'Create dataset for CIFAR-100.',
                                            'create_cifar100 datasets/cifar100 ~/downloads/cifar100')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'cifar100_dir',     help='Directory containing CIFAR-100')

    p = add_command(    'create_svhn',      'Create dataset for SVHN.',
                                            'create_svhn datasets/svhn ~/downloads/svhn')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'svhn_dir',         help='Directory containing SVHN')

    p = add_command(    'create_lsun',      'Create dataset for single LSUN category.',
                                            'create_lsun datasets/lsun-car-100k ~/downloads/lsun/car_lmdb --resolution 256 --max_images 100000')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'lmdb_dir',         help='Directory containing LMDB database')
    p.add_argument(     '--resolution',     help='Output resolution (default: 256)', type=int, default=256)
    p.add_argument(     '--max_images',     help='Maximum number of images (default: none)', type=int, default=None)

    p = add_command(    'create_lsun_wide', 'Create LSUN dataset with non-square aspect ratio.',
                                            'create_lsun_wide datasets/lsun-car-512x384 ~/downloads/lsun/car_lmdb --width 512 --height 384')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'lmdb_dir',         help='Directory containing LMDB database')
    p.add_argument(     '--width',          help='Output width (default: 512)', type=int, default=512)
    p.add_argument(     '--height',         help='Output height (default: 384)', type=int, default=384)
    p.add_argument(     '--max_images',     help='Maximum number of images (default: none)', type=int, default=None)

    p = add_command(    'create_celeba',    'Create dataset for CelebA.',
                                            'create_celeba datasets/celeba ~/downloads/celeba')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'celeba_dir',       help='Directory containing CelebA')
    p.add_argument(     '--cx',             help='Center X coordinate (default: 89)', type=int, default=89)
    p.add_argument(     '--cy',             help='Center Y coordinate (default: 121)', type=int, default=121)

    p = add_command(    'create_from_images', 'Create dataset from a directory full of images.',
                                            'create_from_images datasets/mydataset myimagedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)
    p.add_argument(     'add_condition', help='1 if include labels', type=int, default=0)
    p.add_argument(     'label_file',        help='Label pickle file')

    p = add_command(    'create_from_hdf5', 'Create dataset from legacy HDF5 archive.',
                                            'create_from_hdf5 datasets/celebahq ~/downloads/celeba-hq-1024x1024.h5')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'hdf5_filename',    help='HDF5 archive containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#python dataset_tools.py --create_from_images tfrecord_dir './dataset/' image_dir './Data/LLD-logo_files'
#----------------------------------------------------------------------------
#find ./ -type f | xargs --max-procs=16 -n 5000 \
#    mogrify -resize 128x128\> -extent 128x128\> -gravity center -background white -type TrueColor -colorspace sRGB

#find ./ -type f | xargs --max-procs=16 -n 10000 identify | \
#    fgrep -v " PNG 128x128 128x128+0+0 8-bit sRGB"| cut -d '~' -f 1 | \
#    xargs --max-procs=16 -n 10000 rm
if __name__ == "__main__":
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------
