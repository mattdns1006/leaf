import tensorflow as tf
import pandas as pd
import glob,os,pdb
import matplotlib.pyplot as plt
from sklearn import preprocessing

class Data_loader():
    def __init__(self,in_size,batch_size,n_epochs,clean_df=False,shuffle=True):

        self.in_size = in_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        raw_csv_path = "train.csv"
        train_df = pd.read_csv(raw_csv_path) 

        self.id_by_species = lambda species_name: train_df.loc[train_df['species']== species_name].id.values
        id_to_path = lambda id_no: "images/{0}_preprocessed.jpg".format(id_no)
        self.img_from_id = lambda id_no: imread(id_to_path(id_no))

        self.train_df_clean_path = "train_paths.csv"
        if os.path.exists(self.train_df_clean_path) or clean_df == True:
            train_df['img_path'] = train_df.id.apply(id_to_path)
            self.le = preprocessing.LabelEncoder()
            train_df['species_no'] = self.le.fit_transform(train_df['species'])
            train_df = train_df[['img_path','species','species_no']]
            train_df.to_csv(self.train_df_clean_path,index=0,header=True)
            pdb.set_trace()

    def reader(self):
        csv = tf.train.string_input_producer([self.train_df_clean_path],num_epochs=self.n_epochs)
        reader = tf.TextLineReader(skip_header_lines=1)
        k, v = reader.read(csv)
        defaults = [tf.constant([],dtype = tf.string,shape=[1]),
                    tf.constant([],dtype = tf.string),
                    tf.constant([],dtype = tf.int32,shape=[1])]
        path, species, species_no = tf.decode_csv(v,record_defaults = defaults)
        img = self.getImg(path)
        return [img,path,species] 

    def getImg(self,path):
        image_bytes = tf.read_file(path)
        decoded_img = tf.image.decode_jpeg(image_bytes,channels=1)
        decoded_img = tf.squeeze(tf.image.resize_images(decoded_img,self.in_size))
        return decoded_img

if __name__ == "__main__":
    loader = Data_loader(in_size=[100,100],batch_size=5,n_epochs=2,clean_df=True)
    img,path,species_no = loader.reader()
    out_dir = 'images_test'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with tf.Session() as sess:
        tf.initialize_local_variables().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        count = 1
        try:
            while True:
                img_, path_, species_no_ = sess.run([img,path,species_no])
                out = "Path = {0}, species = {1}. Count = {2}.".format(path_,species_no_,count)
                print(out)
                count += 1
                if count % 100 == 0:
                    save_path = out_dir + "/" + str(count) + ".jpg"
                    plt.imshow(img_)
                    plt.title(out)
                    plt.savefig(save_path)
        except tf.errors.OutOfRangeError:
            print("Finished!")





