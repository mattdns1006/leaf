import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import os,pdb
import load_data
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 0.0001, "Initial learning rate.")
flags.DEFINE_integer("batch_size", 10, "Batch size.")
flags.DEFINE_integer("n_epochs", 100, "Number of training epochs.")
flags.DEFINE_integer("in_h", 108, "Image rows = height.")
flags.DEFINE_integer("in_w", 170, "Image cols = width.")
flags.DEFINE_boolean("load", True, "Load previous checkpoint?")
flags.DEFINE_boolean("train", True, "Training model.")
flags.DEFINE_string("model_path", "model.ckpt", "Save dir.")

class Model():
    def __init__(self,model_path,in_size,batch_size,n_epochs,learning_rate):
        self.model_path = os.path.abspath(os.path.join("models/",model_path))
        self.in_size = in_size
        self.in_h = in_size[0]
        self.in_w = in_size[1]
        self.filter_size = 3
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def graph(self,train=True):
        in_training = train 
        bn = tf.layers.batch_normalization
        with tf.name_scope('input'):
            self.loader = load_data.Data_loader(in_size=self.in_size,batch_size=self.batch_size,n_epochs=self.n_epochs)
            data = self.loader.get_data(train=in_training)
            self.path,self.X,self.Y = data 
            self.X_reshape = tf.reshape(self.X,shape=[-1,self.in_h,self.in_w,1])

        conv1 = tf.layers.conv2d( inputs=self.X_reshape, filters=32, 
            kernel_size=[self.filter_size, self.filter_size], padding="same", activation=tf.nn.relu)
        conv1 = bn(conv1,training=in_training)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

        conv2 = tf.layers.conv2d( inputs=pool1, filters=48,
            kernel_size=[self.filter_size, self.filter_size], padding="same", activation=tf.nn.relu)
        conv2 = bn(conv2,training=in_training)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

        conv3 = tf.layers.conv2d( inputs=pool2, filters=64, 
            kernel_size=[self.filter_size, self.filter_size], padding="same", activation=tf.nn.relu)
        conv3 = bn(conv3,training=in_training)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)

        conv4 = tf.layers.conv2d( inputs=pool3, filters=80, 
            kernel_size=[self.filter_size, self.filter_size], padding="same", activation=tf.nn.relu)
        conv4 = bn(conv4,training=in_training)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3], strides=2)

        conv5 = tf.layers.conv2d( inputs=pool4, filters=96, 
            kernel_size=[self.filter_size, self.filter_size],strides=[2,2], padding="same", activation=tf.nn.relu)
        conv5 = bn(conv5,training=in_training)

        shape = conv5.get_shape().as_list()
        print("Shape at lowest point = {0}".format(shape))
        flat = tf.reshape(conv5, [-1, shape[1]*shape[2]*shape[3]])

        dense = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
        dense = bn(dense,training=in_training)
        #flat = tf.layers.dropout(flat, rate=0.25, training=train)
        dense = tf.layers.dense(inputs=dense, units=256, activation=tf.nn.relu)
        self.logits = tf.layers.dense(inputs=dense, units=99, activation=tf.nn.relu)

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.Y, logits=self.logits)
        predictions = tf.argmax(input=self.logits, axis=1)

        with tf.variable_scope("cm"):
            n_classes = self.loader.le.classes_.size
            cm_diff = tf.confusion_matrix(labels=self.Y,predictions=predictions,num_classes=n_classes)
            cm_init = tf.get_variable("confusion_matrix",[n_classes,n_classes],dtype=tf.int32,
                initializer = tf.zeros_initializer())
            self.cm = tf.assign_add(cm_init, cm_diff)

        self.metrics = {
        "predictions": predictions,
        "probabilities": tf.nn.softmax(self.logits, name="softmax_tensor"),
        "cm": self.cm
        }

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops): #for BN
            self.train_op = self.optimizer.minimize(
                loss=self.loss,
                global_step=self.global_step)

        self.saver = tf.train.Saver()
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Number of trainable parameters = {0}.".format(total_params))


    def session(self,train):
        tf.reset_default_graph()
        with tf.Session() as sess:
            self.graph(train=train)
            coord = tf.train.Coordinator()
            if FLAGS.load == True or train==False: 
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
                        _,loss,path,cm = sess.run([self.train_op,
                        self.loss,self.path,
                        self.metrics['cm']
                        ])
                    else:
                        loss,path = sess.run([self.loss,self.path])
                    count += len(path)
                    losses.append(loss)

                    if count % 500 == 0 and train == True:
                        running_mean = np.array(losses).mean()
                        print("Seen {0}/{1} examples. Losses = {2:.4f}".format(
                        count,
                        self.loader.train_size*self.n_epochs,
                        running_mean
                        ))
                        losses = []
                        self.saver.save(sess,self.model_path)
            except tf.errors.OutOfRangeError:
                print("Finished!")

            if train == False:
                val_loss = np.array(losses).mean()
                print("Test loss = {0:.4f}.".format(val_loss))
                pdb.set_trace()

        sess.close()



if __name__ == "__main__":
    in_size = [FLAGS.in_h,FLAGS.in_w]
    model = Model(model_path=FLAGS.model_path,
            in_size=in_size,
            batch_size=FLAGS.batch_size,
            n_epochs=FLAGS.n_epochs,
            learning_rate=FLAGS.lr)
    model.session(train=True)
    model.session(train=False)

