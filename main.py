import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import os,pdb
import load_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.003, "Initial learning rate.")
flags.DEFINE_integer("batch_size", 20, "Batch size.")
flags.DEFINE_integer("n_epochs", 30, "Number of training epochs.")
flags.DEFINE_integer("in_h", 54, "Image rows = height.")
flags.DEFINE_integer("in_w", 85, "Image cols = width.")
flags.DEFINE_boolean("load", True, "Load previous checkpoint?")
flags.DEFINE_boolean("train", True, "Training model.")
flags.DEFINE_string("model_path", "model.ckpt", "Save dir.")

class Model():
    def __init__(self,model_path,in_size,batch_size,n_epochs,learning_rate):
        self.model_path = os.path.abspath(os.path.join("models/",model_path))
        self.in_size = in_size
        self.in_h = in_size[0]
        self.in_w = in_size[1]
        self.filter_size = 5
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def graph(self,train=True):
        with tf.name_scope('input'):
            self.loader = load_data.Data_loader(in_size=self.in_size,batch_size=self.batch_size,n_epochs=self.n_epochs)
            data = self.loader.get_data(train=train)
            self.X,self.Y = data 
            self.X_reshape = tf.reshape(self.X,shape=[-1,self.in_h,self.in_w,1])

        conv1 = tf.layers.conv2d( inputs=self.X_reshape, filters=32, kernel_size=[self.filter_size, self.filter_size], padding="same", activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

        conv2 = self.conv2 = tf.layers.conv2d( inputs=pool1, filters=32, kernel_size=[self.filter_size, self.filter_size], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

        conv3  = tf.layers.conv2d( inputs=pool2, filters=32, kernel_size=[self.filter_size, self.filter_size], padding="same", activation=tf.nn.relu)

        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)

        pool_flat = tf.reshape(pool3, [-1, 5*9*32])
        dense = tf.layers.dense(inputs=pool_flat, units=32, activation=tf.nn.relu)
        self.logits = tf.layers.dense(inputs=dense, units=99, activation=tf.nn.relu)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.Y, logits=self.logits)

        self.predictions = {
        "classes": tf.argmax(input=self.logits, axis=1),
        "probabilities": tf.nn.softmax(self.logits, name="softmax_tensor")
        }
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(
            loss=self.loss,
            global_step=tf.train.get_global_step())
        self.saver = tf.train.Saver()

    def train(self):
        def session(train):
            tf.reset_default_graph()
            with tf.Session() as sess:
                self.graph(train=train)
                coord = tf.train.Coordinator()
                if FLAGS.load == True: 
                    self.saver.restore(sess,self.model_path)
                else:
                    tf.global_variables_initializer().run()
                tf.local_variables_initializer().run()
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)
                try:
                    count = 0 
                    losses = []
                    while True:
                        if train == True:
                            _,loss,Y = sess.run([self.train_op,self.loss,self.Y])
                        else:
                            loss,Y = sess.run([self.loss,self.Y])
                        count += Y.shape[0]
                        losses.append(loss)
                        if count % 100 == 0:
                            print("Seen {0} examples. Losses = {1:.4f}".format(count,np.array(losses).mean()))
                            losses = []
                            self.saver.save(sess,self.model_path)
                except tf.errors.OutOfRangeError:
                    print("Finished!")
            sess.close()
        #session(train=True)
        session(train=False)

if __name__ == "__main__":
    in_size = [FLAGS.in_h,FLAGS.in_w]
    model = Model(model_path=FLAGS.model_path,
            in_size=in_size,
            batch_size=FLAGS.batch_size,
            n_epochs=FLAGS.n_epochs,
            learning_rate=FLAGS.learning_rate)
    model.train()

    pdb.set_trace()
    print("fin")

