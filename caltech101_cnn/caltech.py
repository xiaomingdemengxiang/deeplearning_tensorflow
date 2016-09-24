#coding=utf-8
import tensorflow as tf
import numpy as np

def shuffle(imgs, labels):
	num = imgs.shape[0]
	perm = np.arange(num)
	np.random.shuffle(perm)
	imgs = imgs[perm]
	labels = labels[perm]
	return imgs,labels

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

images_train = np.load("images_train.npy")
labels_train = np.load("labels_train.npy")
images_test = np.load("images_test.npy")
labels_test = np.load("labels_test.npy")


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 101])

#fisrt conv layer
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

temp = conv2d(x, W_conv1)
h_conv1 = tf.nn.relu(temp + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#second conv layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#densely connected layer
W_fc1 = weight_variable([8*8*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#readout
W_fc2 = weight_variable([1024, 101])
b_fc2 = bias_variable([101])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#train and evaluate the model
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

iterater = 25
batch_count = 78
batch_size = 100
for j in range(iterater):
	images_train, labels_train = shuffle(images_train, labels_train)
	for i in range(batch_count):
		start = i * batch_size
		end = i * batch_size + batch_size
		train_accuracy = accuracy.eval(feed_dict={x:images_train[start:end],y_:labels_train[start:end], keep_prob:1.0})
		print("epoch %d | %d/%d   training accuracy: %g" % (j+1, i+1, batch_count, train_accuracy))
		train_step.run(feed_dict={x:images_train[start:end],y_:labels_train[start:end], keep_prob:0.5})
	
print("test accuracy %g " % accuracy.eval(feed_dict={x:images_test, y_:labels_test, keep_prob:1.0}))

sess.close()


