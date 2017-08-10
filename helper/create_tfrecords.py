import skimage.io as io
import os
from glob import glob
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

root = '/home/saikat/Workspace/Datasets/Chars74K/English/Fnt/'

image_files = glob(os.path.join(root, 'Sample010', '*.png'))
tfrecords_filename = '../data/Chars74K/Sample010.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for img_path in image_files:
    img = io.imread(img_path)

    height = img.shape[0]
    width = img.shape[1]

    img_raw = img.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw)}))

    writer.write(example.SerializeToString())

writer.close()
