import tensorflow as tf
import cv2
import time

img = cv2.imread("../images/input.jpg")
img = cv2.resize(img, (512,512))

img = tf.convert_to_tensor(img, dtype=tf.float32)
img = tf.expand_dims(img, axis=0)

start = time.time()

with tf.device('/GPU:0'):
    gray = tf.image.rgb_to_grayscale(img)
    kernel = tf.ones((5,5,1,1)) / 25.0
    blur = tf.nn.conv2d(gray, kernel, strides=1, padding='SAME')

end = time.time()

print("TensorFlow Time:", end-start)
