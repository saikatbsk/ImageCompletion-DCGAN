import os
import time
import tensorflow as tf
from dcgan import DCGAN

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir',       'checkpoints',           """Path to write logs and checkpoints""")
tf.app.flags.DEFINE_string('images_dir',    'images',                """Path to save generated images""")
tf.app.flags.DEFINE_string('data_dir',      'data',                  """Path to data directory""")
tf.app.flags.DEFINE_integer('max_itr',      10000,                   """Maximum number of iterations""")
tf.app.flags.DEFINE_integer('latest_ckpt',  0,                       """Latest checkpoint timestamp to load""")
tf.app.flags.DEFINE_boolean('is_train',     True,                    """False for generating only""")
tf.app.flags.DEFINE_boolean('is_grayscale', False,                   """True for grayscale images""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 300, """number of examples for train""")

CROP_IMAGE_SIZE = 96

def read_decode(batch_size, f_size):
    files = [os.path.join(FLAGS.data_dir, f) for f in os.listdir(FLAGS.data_dir) if f.endswith('.tfrecords')]
    fqueue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(fqueue)
    features = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)})

    image = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8), tf.float32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image = tf.reshape(image, [height, width, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, CROP_IMAGE_SIZE, CROP_IMAGE_SIZE)
    #image = tf.image.random_flip_left_right(image)

    min_queue_examples = FLAGS.num_examples_per_epoch_for_train
    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    tf.summary.image('images', images)
    return tf.subtract(tf.div(tf.image.resize_images(images, [f_size * 2 ** 4, f_size * 2 ** 4]), 127.5), 1.0)

def main(_):
    dcgan = DCGAN(batch_size=128, f_size=6, z_dim=40,
        gdepth1=216, gdepth2=144, gdepth3=96,  gdepth4=64,
        ddepth1=64,  ddepth2=96,  ddepth3=144, ddepth4=216)

    input_images = read_decode(dcgan.batch_size, dcgan.f_size)

    train_op = dcgan.build(input_images, feature_matching=True)

    g_saver = tf.train.Saver(dcgan.g.variables)
    d_saver = tf.train.Saver(dcgan.d.variables)
    g_checkpoint_path = os.path.join(FLAGS.log_dir, 'g.ckpt')
    d_checkpoint_path = os.path.join(FLAGS.log_dir, 'd.ckpt')
    g_checkpoint_restore_path = os.path.join(
        FLAGS.log_dir, 'g.ckpt-'+str(FLAGS.latest_ckpt))
    d_checkpoint_restore_path = os.path.join(
        FLAGS.log_dir, 'd.ckpt-'+str(FLAGS.latest_ckpt))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # restore or initialize generator
        if os.path.exists(g_checkpoint_restore_path+'.meta'):
            print('Restoring variables:')
            for v in dcgan.g.variables:
                print(' ' + v.name)
            g_saver.restore(sess, g_checkpoint_restore_path)

        if FLAGS.is_train:
            # restore or initialize discriminator
            if os.path.exists(d_checkpoint_restore_path+'.meta'):
                print('Restoring variables:')
                for v in dcgan.d.variables:
                    print(' ' + v.name)
                d_saver.restore(sess, d_checkpoint_restore_path)

            sample_z = sess.run(tf.random_uniform(
                [dcgan.batch_size, dcgan.z_dim], minval=-1.0, maxval=1.0))
            images = dcgan.sample_images(5, 5, inputs=sample_z)

            tf.train.start_queue_runners(sess=sess)
            for itr in range(FLAGS.latest_ckpt+1, FLAGS.max_itr):
                start_time = time.time()
                _, g_loss, d_loss = sess.run(
                    [train_op, dcgan.losses['g'], dcgan.losses['d']])
                duration = time.time() - start_time
                print('step: %d, loss: (G: %.8f, D: %.8f), time taken: %.3f' % \
                    (itr, g_loss, d_loss, duration))

                if itr % 100 == 0:
                    if not os.path.exists(FLAGS.images_dir):
                        os.makedirs(FLAGS.images_dir)

                    filename = os.path.join(FLAGS.images_dir, '%05d.jpg' % itr)
                    with open(filename, 'wb') as f:
                        f.write(sess.run(images))

                    if not os.path.exists(FLAGS.log_dir):
                        os.makedirs(FLAGS.log_dir)

                    g_saver.save(sess, g_checkpoint_path, global_step=itr)
                    d_saver.save(sess, d_checkpoint_path, global_step=itr)
        else:
            generated = sess.run(dcgan.sample_images(5, 5))

            if not os.path.exists(FLAGS.images_dir):
                os.makedirs(FLAGS.images_dir)

            filename = os.path.join(FLAGS.images_dir, 'generated_image.jpg')
            with open(filename, 'wb') as f:
                print('write to %s' % filename)
                f.write(generated)

if __name__ == '__main__':
    tf.app.run()
