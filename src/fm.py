'''
Tensorflow implementation of Factorization Machines (FM) as described in:
Xiangnan He, Tat-Seng Chua. Neural Factorization Machines for Sparse Predictive Analytics. In Proc. of SIGIR 2017.

Note that the original paper of FM is: Steffen Rendle. Factorization Machines. In Proc. of ICDM 2010.

@author:
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)

@references:
'''
import math
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss, roc_auc_score
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='../../output/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pnn',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=10,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=1,
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=.1, #0.8
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='GradientDescentOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')

    return parser.parse_args()


class FM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, pretrain_flag, save_file, hidden_factor, loss_type, epoch, batch_size, learning_rate,
                 lamda_bilinear, keep,
                 optimizer_type, batch_norm, verbose, random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.loss_type = loss_type
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features) # 克赛vixi
            self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)  # None * K
            # get the element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            print('nonzero_embeddings', nonzero_embeddings)

            # _________ square_sum part _____________
            self.squared_features_emb = tf.square(nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep)  # dropout at the FM layer

            # _________out _________
            Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features),
                                              1)  # None * 1,WiXi
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1,W0
            self.out = tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1

            # Compute the loss.
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(
                        tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                self.out = tf.sigmoid(self.out)
                if self.lamda_bilinear > 0:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, epsilon=1e-07,
                                                           scope=None) + tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, epsilon=1e-07, scope=None)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_linear_weights = pretrain_graph.get_tensor_by_name('linear_weights:0')
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                fe, fb, b, l = sess.run([feature_embeddings, feature_bias, bias, feature_linear_weights])
            all_weights['linear_weights'] = tf.Variable(l, dtype=tf.float32)
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else:
            all_weights['linear_weights'] = tf.Variable(
                tf.random_normal([self.features_M, 1], 0.0, 0.001),
                name='linear_weights')  # features_M * K
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
                name='feature_embeddings')  # features_M * K
            all_weights['feature_bias'] = tf.Variable(
                tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1
            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        # print(self.sess.run(self.weights, feed_dict=feed_dict))
        # print('loss', loss)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    def get_next_block_from_data(self, data, start_index=0, batch_size=-1):  # generate a random block of training data
        # start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break

        return {'X': X, 'Y': Y}


    def shuffle_in_unison_scary(self, a, b):  # shuffle two lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, Train_data, Test_data):  # fit a dataset
        # Check Init performance
        # if self.verbose > 0:
        #     t2 = time()
        #     init_train, train_auc ,train_rmse= self.evaluate(Train_data,epoch)
        #     init_valid, valid_auc ,valid_rmse = self.evaluate(Validation_data,epoch)
        #
        #     print("Init: \t train_loss=%.8f\tvalid_loss=%.8f\ttrain_auc=%.8f\tvalid_auc=%.8f\t[%.1f s]" % (
        #         init_train, init_valid, train_auc, valid_auc, time() - t2))

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            train_result, train_auc ,train_rmse= self.evaluate(Train_data, epoch)
            # train_result, train_auc = 0, 0
            valid_result, valid_auc ,valid_rmse= self.evaluate(Test_data, epoch)

            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)

            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain_loss=%.8f\tvalid_loss=%.8f\ttrain_auc=%.8f\tvalid_auc=%.8f\t[%.1f s]"
                      % (epoch + 1, t2 - t1, train_result, valid_result, train_auc, valid_auc, time() - t2))
            if self.eva_termination(self.valid_rmse):
                break

        if self.pretrain_flag < 0:
            print("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        else:
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        return False

    # def evaluate(self, data):  # evaluate the results for an input set
    #     num_example = len(data['Y'])
    #     feed_dict = {self.train_features: data['X'], self.train_labels: [[y] for y in data['Y']],
    #                  self.dropout_keep: 1.0, self.train_phase: False}
    #     predictions = self.sess.run((self.out), feed_dict=feed_dict)
    #     y_pred = np.reshape(predictions, (num_example,))
    #     y_true = np.reshape(data['Y'], (num_example,))
    #     auc = roc_auc_score(y_true, y_pred)
    #     if self.loss_type == 'square_loss':
    #         predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
    #         predictions_bounded = np.minimum(predictions_bounded,
    #                                          np.ones(num_example) * max(y_true))  # bound the higher values
    #         RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
    #         return RMSE, auc
    #     elif self.loss_type == 'log_loss':
    #         logloss = log_loss(y_true, y_pred)
    #         return logloss, auc

    def evaluate(self, data,epoch):  # evaluate the results for an input set
        num_example = len(data['Y'])
        size = 3000
        total_batch = int(num_example / size)
        preds = []
        for i in range(total_batch + 1):
            batch_xs = self.get_next_block_from_data(data, size * i, size)
            feed_dict = {self.train_features: batch_xs['X'], self.train_labels: [y for y in batch_xs['Y']],
                         self.dropout_keep: 1.0, self.train_phase: False}
            predictions = self.sess.run((self.out), feed_dict=feed_dict)
            preds.extend(list(predictions))
        # print(preds[:5])
        y_pred = np.reshape(preds, (num_example,))

        '''
        ################################################################################
        with open('../../output/vote_f&p_nn/submission_fm_{}'.format(epoch), 'w') as outfile:
            for index, item in enumerate(y_pred, start=0):
                outfile.write(str(index) + ',' + str(item) + '\n')
        ################################################################################
        '''
        y_true = np.reshape(data['Y'], (num_example,))
        auc = roc_auc_score(y_true, y_pred)
        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded,
                                         np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        logloss = log_loss(y_true, y_pred)
        return logloss, auc, RMSE
        # if self.loss_type == 'square_loss':
        #     predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        #     predictions_bounded = np.minimum(predictions_bounded,
        #                                      np.ones(num_example) * max(y_true))  # bound the higher values
        #     RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        #     logloss = log_loss(y_true, y_pred)
        #     return RMSE, auc,logloss
        # elif self.loss_type == 'log_loss':
        #     logloss = log_loss(y_true, y_pred)
        #     return logloss, auc



if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)
    if args.verbose > 0:
        print(
            "FM: dataset=%s, factors=%d, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
            % (args.dataset, args.hidden_factor, args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda,
               args.keep_prob, args.optimizer, args.batch_norm))

    save_file = '../../output/pnn/pretrain/%s_%d' % (args.dataset, args.hidden_factor)
    # Training
    t1 = time()
    model = FM(data.features_M, args.pretrain, save_file, args.hidden_factor, args.loss_type, args.epoch,
               args.batch_size, args.lr, args.lamda, args.keep_prob, args.optimizer, args.batch_norm, args.verbose)
    model.train(data.Train_data, data.Test_data)

    # Find the best validation result across iterations
    best_valid_score = 0
    if args.loss_type == 'square_loss':
        best_valid_score = min(model.valid_rmse)
    elif args.loss_type == 'log_loss':
        best_valid_score = max(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.test_rmse[best_epoch],
             time() - t1))
