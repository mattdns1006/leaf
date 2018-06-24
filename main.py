import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing 
from sklearn.decomposition import PCA
import pdb 
import tensorflow as tf

_PCA = True
N_COMP = 40 
N_ITER = 5 # Number of re-shuffling splitting iterations
DIM = N_COMP if _PCA else 192
N_SPECIES = 99
LR = 2
BETA = 0.000001
BS = 30
N_EPOCHS = 100
TEST_SIZE = 0.2
WRITE_TEST = False

# Read
X = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
Y = X['species']
X.drop(['id','species'],axis=1,inplace=1)
test_id = test['id']
test.drop(['id'],axis=1,inplace=1)

#pca
pca = PCA(n_components=N_COMP)
X = pca.fit_transform(X).astype(np.float32)
test = pca.transform(test).astype(np.float32)

# encode Y
lenc = preprocessing.LabelEncoder()
Y_idx = lenc.fit_transform(Y)
#X[:,0] = (Y - Y.mean())/Y.std() # Sanity
Y = np.eye(N_SPECIES)[Y_idx].astype(np.float32)

# Graph
x = tf.placeholder(tf.float32,[None,N_COMP])
y = tf.placeholder(tf.float32,[None,N_SPECIES])
weights = {
    'w1': tf.Variable(tf.random_normal([N_COMP, N_SPECIES])),
}
biases = {
    'b1': tf.Variable(tf.random_normal([N_SPECIES])),
}
logits = tf.add(tf.matmul(x,weights['w1']),biases['b1'])

# loss functions
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
reg = tf.nn.l2_loss(weights['w1'])
loss_op = tf.reduce_mean(loss + BETA* reg)
optimizer = tf.train.AdamOptimizer(learning_rate=LR)
train_op = optimizer.minimize(loss_op)

#Predictions
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
predict = tf.nn.softmax(logits)

sss = StratifiedShuffleSplit(Y_idx, N_ITER, test_size=TEST_SIZE, random_state=0)
with tf.Session() as sess:

    print("Print explained var = {0:.3f}".format(pca.explained_variance_ratio_.sum()))
    fold = 0 
    for train_index, test_index in sss:
        init = tf.global_variables_initializer()
        sess.run(init)
        if WRITE_TEST == True:
            X_train, Y_train = X, Y
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        for step in range(1, N_EPOCHS+1):
            batch_x, batch_y = X_train, Y_train
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        tr_loss, tr_acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Final loss= " + "{:.4f}".format(tr_loss) + ", Training Accuracy= " + "{:.3f}".format(tr_acc))

        if WRITE_TEST == True:
            break
        te_loss, te_acc = sess.run([loss, accuracy], feed_dict={x: X_test, y: Y_test})
        print("Final loss= " + "{:.4f}".format(te_loss) + ", Testing Accuracy= " + "{:.3f}".format(te_acc))
        print("Finished {0} of {1} folds.".format(fold,N_ITER))
        fold += 1

    if WRITE_TEST == True:
        head = lenc.inverse_transform(np.arange(0,N_SPECIES))
        test_predictions = predict.eval(feed_dict={x: test})
        csv = pd.DataFrame(test_predictions,columns=head)
        csv['id'] = test_id
        csv.to_csv("preds.csv",index=0)
        print("Test preds written")










