import tensorflow as tf
import pandas as pd
import glob,os,pdb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit as SSS
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

class Data_loader():
    def __init__(self,in_training,in_size,batch_size,n_epochs,aug_flip,clean_df=False):
        self.in_training = in_training
        self.in_size = in_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.aug_flip = aug_flip

        raw_csv_path = "train.csv"
        df = pd.read_csv(raw_csv_path) 

        self.id_by_species = lambda species_name: train_df.loc[train_df['species']== species_name].id.values
        id_to_path = lambda id_no: "images/{0}_preprocessed.jpg".format(id_no)
        self.img_from_id = lambda id_no: imread(id_to_path(id_no))

        self.train_df_clean_path = "train_paths.csv"
        self.val_df_clean_path = "val_paths.csv"
        if os.path.exists(self.train_df_clean_path) or clean_df == True:
            df['img_path'] = df.id.apply(id_to_path)
            self.le = preprocessing.LabelEncoder()
            df['species_no'] = self.le.fit_transform(df['species'])
            df = df[['img_path','species','species_no']]
            # split into train and validation set
            sss = SSS(y=df.species_no,n_iter = 1,test_size=0.2,random_state=1006)
            for train_index, val_index in sss:
                train_df, val_df = df.ix[train_index], df.ix[val_index]

            train_df.to_csv(self.train_df_clean_path,index=0,header=True)
            val_df.to_csv(self.val_df_clean_path.replace("train","val"),index=0,header=True)
        num_lines = lambda path: pd.read_csv(path).shape[0] 
        self.train_size = num_lines(self.train_df_clean_path)
        self.val_size = num_lines(self.val_df_clean_path)

    def reader(self,csv_path):
        n_epochs = self.n_epochs if self.in_training == True else 1
        batch_size = self.batch_size if self.in_training else 1
        csv = tf.train.string_input_producer([csv_path],num_epochs=n_epochs)
        reader = tf.TextLineReader(skip_header_lines=1)
        k, v = reader.read(csv)
        defaults = [tf.constant([],dtype = tf.string,shape=[1]),
                    tf.constant([],dtype = tf.string),
                    tf.constant([],dtype = tf.int32,shape=[1])]
        path, species, species_no = tf.decode_csv(v,record_defaults = defaults)
        img = self.getImg(path)
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3*batch_size
        path_batch, img_batch, label_batch = tf.train.shuffle_batch([path, img, species_no], 
                batch_size=batch_size,
                capacity=capacity, 
                min_after_dequeue=min_after_dequeue,
                allow_smaller_final_batch=True)
        return [path_batch,img_batch,label_batch] 

    def getImg(self,path):
        image_bytes = tf.read_file(path)
        decoded_img = tf.image.decode_jpeg(image_bytes,channels=1)
        decoded_img = tf.cast(decoded_img,tf.float32)
        decoded_img = tf.multiply(decoded_img,1/255.0)
        if self.in_training == True and self.aug_flip == True:
            print("\n !!!Doing image flips!!!\n")
            decoded_img = tf.image.random_flip_left_right(decoded_img)
            decoded_img = tf.image.random_flip_up_down(decoded_img)
            print("\n !!!Doing rotations !!!\n")
            angle = tf.random_normal(mean=0,stddev=0.19,shape=[1])
            decoded_img = tf.contrib.image.rotate(decoded_img,angles=angle)
        decoded_img = tf.squeeze(tf.image.resize_images(decoded_img,self.in_size))
        return decoded_img

    def get_data(self):
        if self.in_training == True:
            return self.reader(self.train_df_clean_path)
        else:
            return self.reader(self.val_df_clean_path)

if __name__ == "__main__":

    def session(loader):
        data =loader.get_data()
        with tf.Session() as sess:
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            count = 0 
            try:
                while True:
                    data_ = sess.run([data],feed_dict={loader.k:0})[0]
                    count += data_[0].shape[0]
            except tf.errors.OutOfRangeError:
                print("\n Finished! Seen {0} examples.\n".format(count))
        sess.close()

    #train_loader = Data_loader(in_training=True,in_size=[68,106],batch_size=5,n_epochs=5,clean_df=True,aug_flip=True)
    #session(data=train_loader.get_data())

    test_loader = Data_loader(in_training= False,in_size=[68,106],batch_size=5,n_epochs=5,clean_df=True,aug_flip=False)
    session(test_loader)
    pdb.set_trace()


