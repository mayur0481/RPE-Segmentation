import tensorflow as tf


def im_parser(shape = [256, 256]):
    def parser(f1, f2):
        image = tf.io.read_file(f1)
        image = tf.io.decode_jpeg(image)
        image = image[:, :, 0:1]
        image = tf.image.resize(image, shape, antialias = True)
        image = tf.cast(image, tf.float32) 


        label = tf.io.read_file(f2)
        label = tf.io.decode_png(label)
        label = tf.image.resize(label, shape, method = 'nearest')

        return image, label
    
    return parser






