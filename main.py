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
flags.DEFINE_boolean("aug_flip", True, "Augmentation with random flips")
flags.DEFINE_integer("batch_size", 10, "Batch size.")
flags.DEFINE_integer("n_epochs", 100, "Number of training epochs.")
flags.DEFINE_integer("in_h", 108, "Image rows = height.")
flags.DEFINE_integer("in_w", 170, "Image cols = width.")
flags.DEFINE_boolean("load", True, "Load previous checkpoint?")
flags.DEFINE_boolean("train", True, "Training model.")
flags.DEFINE_string("model_path", "model.ckpt", "Save dir.")
flags.DEFINE_string("summaries_dir", "models/summaries/", "Summaries directory.")

class Model():
    def __init__(self,model_path,in_size,batch_size,n_epochs,learning_rate,aug_flip):
        self.model_path = os.path.abspath(os.path.join("models/",model_path))
        self.in_size = in_size
        self.in_h = in_size[0]
        self.in_w = in_size[1]

        self.aug_flip = aug_flip 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.filter_size = 3
        self.n_filters = 32
        self.n_filters_inc = 16
        self.n_layers = 5
        self.pool_size = 3

    def graph(self,in_training=True):
        shape_name = lambda tensor: print("Tensor {0} has shape = {1}.\n".format(tensor.name,tensor.get_shape().as_list()))

        with tf.name_scope('input'):
            self.loader = load_data.Data_loader(in_training=in_training,in_size=self.in_size,batch_size=self.batch_size,n_epochs=self.n_epochs,aug_flip=self.aug_flip)
            data = self.loader.get_data()
            self.path,self.X,self.Y = data 
            self.X_reshape = tf.reshape(self.X,shape=[-1,self.in_h,self.in_w,1])

        for layer_no in range(1,self.n_layers+1):
            if layer_no == 1:
                in_tensor = self.X_reshape 
                n_filters = self.n_filters
            else: 
                in_tensor = out_tensor
                n_filters += self.n_filters_inc

            conv = tf.layers.conv2d( inputs=in_tensor, 
                filters=n_filters, 
                kernel_size=[self.filter_size, self.filter_size], 
                padding="same", 
                activation=tf.nn.relu, 
                name = "conv_{0}".format(layer_no))

            conv_bn = tf.layers.batch_normalization(conv,training=in_training,
                name="bn_{0}".format(layer_no))

            pool = tf.layers.max_pooling2d(inputs=conv_bn, 
                pool_size=[self.pool_size, self.pool_size], 
                strides=2,
                name="pool_{0}".format(layer_no)
                )
            [shape_name(tensor) for tensor in [conv,conv_bn,pool]]
            out_tensor = pool

        shape = out_tensor.get_shape().as_list()
        print("Shape at lowest point = {0}".format(shape))
        flat = tf.reshape(out_tensor, [-1, shape[1]*shape[2]*shape[3]])

        dense = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)
        dense = tf.layers.batch_normalization(dense,training=in_training)
        #flat = tf.layers.dropout(flat, rate=0.25, training=train)
        #dense = tf.layers.dense(inputs=dense, units=256, activation=tf.nn.relu)
        self.logits = tf.layers.dense(inputs=dense, units=99, activation=tf.nn.relu)

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.Y, logits=self.logits)
        predictions = tf.argmax(input=self.logits, axis=1)

        with tf.variable_scope("cm"):
            n_classes = self.loader.le.classes_.size
            cm_diff = tf.confusion_matrix(labels=self.Y,predictions=predictions,num_classes=n_classes)
            cm_init = tf.get_variable("confusion_matrix",[n_classes,n_classes],dtype=tf.int32,
                initializer = tf.zeros_initializer())
            self.cm = tf.assign_add(cm_init, cm_diff)

        correct_prediction = tf.equal(tf.cast(self.Y,tf.int64),predictions)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        self.metrics = {
        "predictions": predictions,
        "probabilities": tf.nn.softmax(self.logits, name="softmax_tensor"),
        "cm": self.cm,
        "acc":accuracy
        }


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops): #for BN
            self.train_op = self.optimizer.minimize(
                loss=self.loss,
                global_step=self.global_step)

        self.saver = tf.train.Saver()
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Number of trainable parameters = {0}.".format(total_params))

        tf.summary.scalar('acc', self.metrics['acc'])
        tf.summary.scalar('loss',self.loss)
        self.merged = tf.summary.merge_all()


    def session(self,train):
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            self.graph(in_training=train)
            train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
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
                        _,summary,loss,path,cm,acc = sess.run([
                        self.train_op,
                        self.merged,
                        self.loss,
                        self.path,
                        self.metrics['cm'],
                        self.metrics['acc']
                        ])
                        train_writer.add_summary(summary,tf.train.global_step(sess,self.global_step))
                    else:
                        summary,loss,path,cm,acc = sess.run([
                        self.merged,
                        self.loss,
                        self.path,
                        self.metrics['cm'],
                        self.metrics['acc']
                        ])
                        test_writer.add_summary(summary,tf.train.global_step(sess,self.global_step))
                    count += len(path)
                    losses.append(loss)

                    if count % 500 == 0 and train == True:
                        running_mean = np.array(losses).mean()
                        print("Seen {0}/{1} examples. Losses = {2:.4f}. Acc = {3:.4f}. in_training = {4}.".format(
                        count,
                        self.loader.train_size*self.n_epochs,
                        running_mean,
                        acc,
                        self.loader.in_training
                        ))
                        losses = []
                        self.saver.save(sess,self.model_path)
            except tf.errors.OutOfRangeError:
                print("Finished!")

            if train == False:
                val_loss = np.array(losses).mean()
                print("Test loss = {0:.4f}. Acc = {1:.4f}.".format(val_loss,acc))
                pdb.set_trace()

        sess.close()



if __name__ == "__main__":
    in_size = [FLAGS.in_h,FLAGS.in_w]
    model = Model(model_path=FLAGS.model_path,
            in_size=in_size,
            batch_size=FLAGS.batch_size,
            n_epochs=FLAGS.n_epochs,
            learning_rate=FLAGS.lr,
            aug_flip=FLAGS.aug_flip)
    model.session(train=True)
    model.session(train=False)

