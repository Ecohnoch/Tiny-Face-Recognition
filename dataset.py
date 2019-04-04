import tensorflow as tf 

def parse_function(example_proto):
	features = {'image_raw': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64)}
	features = tf.parse_single_example(example_proto, features)

	img = tf.image.decode_jpeg(features['image_raw'])
	img = tf.reshape(img, shape=(112, 112, 3))
	r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
	img = tf.concat([b, g, r], axis=-1)
	img = tf.cast(img, dtype=tf.float32)
	img = tf.subtract(img, 127.5)
	img = tf.multiply(img, 0.0078125)
	img = tf.image.random_flip_left_right(img)
	label = tf.cast(features['label'], tf.int64)
	return img, label


tfr = 'tfrecords/tran.tfrecords'
dataset = tf.data.TFRecordDataset(tfr)
dataset = dataset.map(parse_function)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	sess.run(iterator.initializer)

	while True:
		image_train, label_train = sess.run(next_element)
		print(image_train.shape, label_train.shape) 
		print(label_train)