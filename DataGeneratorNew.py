import tensorflow as tf
from ImageParser import im_parser
from ImageAugment import _augment
#from sklearn.model_selection import train_test_split
'''
def retina_data(x_path, y_path, batch_size = 32, center = True, augment = True, hfp = 0.5, vfp = 0.5, shape = [500, 500], rep_method = 'bilinear'):
    x_paths = tf.io.gfile.glob(x_path)
    y_paths = tf.io.gfile.glob(y_path)
    
    x_train, x_val, y_train, y_val = train_test_split(x_paths, y_paths, test_size = 0.3)
    
    x_train_paths = tf.data.Dataset.list_files(x_train, shuffle = False)
    y_train_paths = tf.data.Dataset.list_files(y_train, shuffle = False)
    
    x_val_paths = tf.data.Dataset.list_files(x_val, shuffle = False)
    y_val_paths = tf.data.Dataset.list_files(y_val, shuffle = False)

    train_data = tf.data.Dataset.zip((x_train_paths, y_train_paths))
    val_data = tf.data.Dataset.zip((x_val_paths, y_val_paths))

    train_data = train_data.shuffle(500)
    val_data = val_data.shuffle(500)
    train_data = train_data.map(im_parser)
    val_data = val_data.map(im_parser)

    if augment:
        train_data = train_data.map(_augment(hfp, vfp))

   
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    
    train_data = train_data.map(lambda im, lab : (tf.image.resize(im, shape, method = rep_method, antialias = True), tf.image.resize(lab, shape, method = 'nearest')))
    
    val_data = val_data.map(lambda im, lab : (tf.image.resize(im, shape, method = rep_method, antialias = True), tf.image.resize(lab, shape, method = 'nearest')))
    

    if center:
        train_data = train_data.map(lambda im, lab : (im * 1./127.5 - 1, lab))
        val_data = val_data.map(lambda im, lab : (im * 1./127.5 - 1, lab))
    else:
        train_data = train_data.map(lambda im, lab : (im * 1./255.0, lab))
        val_data = val_data.map(lambda im, lab : (im * 1./255.0, lab))
       
    train_data = train_data.repeat()
    val_data = val_data.repeat()

    return train_data, val_data
'''


def retina_data1(x_train, y_train, x_val, y_val,  batch_size = 32, center = True, augment = False, hfp = 0.5, vfp = 0.1, shape = [256, 256]):
    x_train_paths = tf.data.Dataset.list_files(x_train, shuffle = False)
    y_train_paths = tf.data.Dataset.list_files(y_train, shuffle = False)

    x_val_paths = tf.data.Dataset.list_files(x_val, shuffle = False)
    y_val_paths = tf.data.Dataset.list_files(y_val, shuffle = False)

    train_data = tf.data.Dataset.zip((x_train_paths, y_train_paths))
    val_data = tf.data.Dataset.zip((x_val_paths, y_val_paths))

    train_data = train_data.shuffle(350)
    
    train_data = train_data.map(im_parser(shape))
    val_data = val_data.map(im_parser(shape))
    
    if augment:
        train_data = train_data.map(_augment(hfp, vfp))

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    if center:
        train_data = train_data.map(lambda im, lab : (im * 1./127.5 - 1, lab))
        val_data = val_data.map(lambda im, lab : (im * 1./127.5 - 1, lab))

    else:
        train_data = train_data.map(lambda im, lab : (im * 1./255.0, lab))
        val_data = val_data.map(lambda im, lab : (im * 1./255.0, lab))


    train_data = train_data.repeat()
    val_data = val_data.repeat()


    return train_data, val_data


def retina_test(x_test, y_test, batch_size = 32, center = True, shape = [256, 256]):
    x_test_paths = tf.data.Dataset.list_files(x_test, shuffle = False)
    y_test_paths = tf.data.Dataset.list_files(y_test, shuffle = False)

    test_data = tf.data.Dataset.zip((x_test_paths, y_test_paths))
    
    test_data = test_data.map(im_parser(shape))

    test_data = test_data.batch(batch_size)


    if center:
        test_data = test_data.map(lambda im, lab : (im * 1./127.5 - 1, lab))
    else:
        test_data = test_data.map(lambda im, lab : (im * 1./255.0, lab))



    test_data = test_data.repeat()

    return test_data


