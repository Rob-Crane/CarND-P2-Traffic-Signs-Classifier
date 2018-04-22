# Load pickled data
import pickle
import tensorflow as tf
from tensorflow.python import debug as tf_debug

# training_file = 'train.p'
# validation_file = 'valid.p'
# testing_file = 'test.p'

# with open(training_file, mode='rb') as f:
    # train = pickle.load(f)
# with open(validation_file, mode='rb') as f:
    # valid = pickle.load(f)
# with open(testing_file, mode='rb') as f:
    # test = pickle.load(f)
                            
# X_train, y_train = train['features'], train['labels']
# X_valid, y_valid = valid['features'], valid['labels']
# X_test, y_test = test['features'], test['labels']

# Copied from Udacity
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify = y_train)

n_train = len(X_train)

IMAGE_SIZE=24

def get_train_batch(images=X_train, labels=y_train, batch_size=32):
    """Specify input to model
    
    Args:
        images: numpy.ndarray of images
        labels: numpy.ndarray of labels
        batch_size: number of examples to batch in model
        
    Returns:
        """
    assert images.shape[0] == labels.shape[0]
    with tf.name_scope('image_preprocessing'):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(n_train)
        def _preprocess_image(image):
            image = tf.cast(image, tf.float32)
            image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            image = tf.image.per_image_standardization(image)
            return image
        dataset = dataset.map(lambda image,label: (_preprocess_image(image), tf.cast(label, tf.int32)))
        dataset = dataset.batch(batch_size).repeat()
        return dataset.make_one_shot_iterator().get_next()

def get_validation_batch(images=X_valid, labels=y_valid, batch_size=32):
    return _get_evaluation_data(images, labels, batch_size)

def get_test_batch(images=X_test, labels=y_test, batch_size=32):
    return _get_evaluation_data(images, labels, batch_size)

def _get_evaluation_data(images, labels, batch_size):
    assert images.shape[0] == labels.shape[0]
    with tf.name_scope('validation_preprocessing'):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(len(labels)).repeat()
        def _preprocess_image(image):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize_image_with_crop_or_pad(
                    image,
                    IMAGE_SIZE,
                    IMAGE_SIZE)
            image = tf.image.per_image_standardization(image)
            return image
        dataset = dataset.map(lambda image,label: (_preprocess_image(image), tf.cast(label, tf.int32)))
        dataset = dataset.batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
