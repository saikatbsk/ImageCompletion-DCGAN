import os
import time
import tensorflow as tf
from utils import *
from dcgan import DCGAN

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir',       'checkpoints', """Path to write logs and checkpoints""")
tf.app.flags.DEFINE_string('images_dir',    'images',      """Path to save generated images""")
tf.app.flags.DEFINE_string('data_dir',      'data',        """Path to data directory""")
tf.app.flags.DEFINE_integer('max_itr',      10000,         """Maximum number of iterations""")
tf.app.flags.DEFINE_integer('latest_ckpt',  0,             """Latest checkpoint timestamp to load""")
tf.app.flags.DEFINE_boolean('is_train',     True,          """False for generating only""")
tf.app.flags.DEFINE_boolean('is_grayscale', False,         """True for grayscale images [not yet implemented]""")

def main(_):
    dcgan = DCGAN(batch_size=128, f_size=6, z_dim=40,
        gdepth1=216, gdepth2=144, gdepth3=96,  gdepth4=64,
        ddepth1=64,  ddepth2=96,  ddepth3=144, ddepth4=216)

    """
    Batch training not implemented yet. Keep exactly 128 images inside data_dir.
    """
    input_images = input_data(FLAGS.data_dir,
        input_height=128, input_width=128,
        resize_height=96, resize_width=96,
        is_grayscale=FLAGS.is_grayscale)
    print('Images loaded from: ', FLAGS.data_dir)
    print('Input shape: ', input_images.shape)

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
