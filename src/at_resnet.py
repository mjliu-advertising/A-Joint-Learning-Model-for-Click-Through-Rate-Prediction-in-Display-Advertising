
import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time
import argparse
import LoadData as DATA
#from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


# tf.random_seed = 2019
# np.random_seed = 2019


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--path', nargs='?', default='../../output/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pnn',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--attention', type=int, default=1,
                        help='flag for attention. 1: use attention; 0: no attention')
    parser.add_argument('--lamda_attention', type=float, default=1e+2,
                        help='Regularizer for attention part.')
    parser.add_argument('--hidden_factor', nargs='?', default='[10,10]',
                        help='Number of hidden factors.')
    parser.add_argument('--valid_dimen', type=int, default=16,
                        help='Valid dimension of the dataset. (e.g. ipinyou=16, criteo=34)')
    parser.add_argument('--layers_1', nargs='?', default='[500, 500, 500, 500, 500, 500]',
                        help="Size of each layer.")
    parser.add_argument('--layers_2', nargs='?', default='[500, 500, 500, 500]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[.9, .9, .9, .9, .9, .9, .9]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0.000,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='gd',
                        help='Specify an optimizer type (adam, adagrad, gd, ftrl, adaldeta, padagrad, rmsprop, pgd, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity, elu')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()


class NeuralFM(BaseEstimator, TransformerMixin):
    def __init__(self, field_cnt, features_M, hidden_factor, valid_dimension, attention, lamda_attention, layers_1, layers_2, loss_type,
                 pretrain_flag, epoch, batch_size,
                 learning_rate,
                 lamda_bilinear,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop, random_seed=2017):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.valid_dimension = valid_dimension
        self.attention = attention
        self.lamda_attention = lamda_attention
        self.layers_1 = layers_1
        self.layers_2 = layers_2
        self.loss_type = loss_type
        self.pretrain_flag = pretrain_flag
        self.field_cnt = field_cnt
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.early_stop = early_stop
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        self.num_pairs = (self.field_cnt)* (self.field_cnt - 1) / 2
        self.nn_input_dim = self.field_cnt * self.hidden_factor[0] + (self.field_cnt)* (self.field_cnt - 1) // 2

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

            # Input data
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # get the summed up embeddings of features.
            self.nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features) #

            w_fm_nn_input = tf.reshape(self.nonzero_embeddings, [-1, self.field_cnt * self.hidden_factor[1]])  #


            # element-wise product
            element_wise_product_list = []
            count = 0
            for i in range(0, self.valid_dimension):  # filed_size
                for j in range(i + 1, self.valid_dimension):
                    element_wise_product_list.append(
                        tf.multiply(self.nonzero_embeddings[:, i, :], self.nonzero_embeddings[:, j, :]))
                    count += 1

            self.element_wise_product = tf.stack(element_wise_product_list)
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2],
                                                     name="element_wise_product")


            # _________ One MLP Layer / attention part _____________
            num_interactions = int(self.valid_dimension * (self.valid_dimension - 1) / 2)
            if self.attention:
                self.attention_mul = tf.reshape(
                    tf.matmul(tf.reshape(self.element_wise_product, shape=[-1, self.hidden_factor[1]]),
                              # 此为embedding_size K
                              self.weights['attention_W']),
                    shape=[-1, num_interactions, self.hidden_factor[0]])

                self.attention_relu = tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(self.attention_mul
                                         + self.weights['attention_b'])), 2, keep_dims=True)           # N*(F*(F-1)/2)*1

                self.attention_out = tf.nn.softmax(self.attention_relu) # N*(F*(F-1)/2)*1
                # self.attention_out = tf.nn.dropout(self.attention_out,
                #                                    self.dropout_keep[0])


            # _________ Attention-aware Pairwise Interaction Layer _____________
            if self.attention:
                self.at_out = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product), 2,
                                         name="afm")  # None * (F*(F-1)/2)

            else:
                self.out = tf.reduce_sum(self.element_wise_product, 2, name="afm")
            #self.AFM_FM = tf.reduce_sum(self.element_wise_product, 2, name="afm_fm")
            #self.AFM_FM = self.AFM_FM / num_interactions

            self.at_out = tf.nn.dropout(self.at_out, self.dropout_keep[5])  # dropout

            w_fm_nn_input = tf.concat([w_fm_nn_input, self.at_out], 1)

            print('w_fm_nn_input.shape:', w_fm_nn_input.shape)


            # resnet_left
            print('---the left part---')
            self.FM = w_fm_nn_input
            for i in range(0, 2):
                self.FM = tf.add(tf.matmul(self.FM, self.weights['left_layer_%d' % i]),
                                 self.weights['left_bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                    scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.FM = self.activation_function(self.FM)
                self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i])  # dropout at each Deep layer
                if i == 0:
                    self.layer1out = self.FM

            self.FM = tf.add(tf.matmul(self.FM, self.weights['left_layer_2']),
                             self.weights['left_bias_2'])
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                scope_bn='bn_%d' % i)  # None * layer[i] * 1
            self.FM = tf.add(self.FM, self.layer1out)
            self.FM = self.activation_function(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[2])

            for i in range(3, 5):
                self.FM = tf.add(tf.matmul(self.FM, self.weights['left_layer_%d' % i]),
                                 self.weights['left_bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                    scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.FM = self.activation_function(self.FM)
                self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i])  # dropout at each Deep layer
                print('i:', i)
                if i == 3:
                    self.layer4out = self.FM

            self.FM = tf.add(tf.matmul(self.FM, self.weights['left_layer_5']),
                             self.weights['left_bias_5'])
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                scope_bn='bn_%d' % i)  # None * layer[i] * 1
            self.FM = tf.add(self.FM, self.layer4out)
            self.FM = self.activation_function(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[5])
            leftout = self.FM
            print('leftout.shape:', leftout.shape)


           
           
            # resnet_right
            print('---the right part---')
            self.FM = w_fm_nn_input
            for i in range(0, 2):

                self.FM = tf.add(tf.matmul(self.FM, self.weights['right_layer_%d' % i]),
                                 self.weights['right_bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                    scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.FM = self.activation_function(self.FM)
                self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i])  # dropout at each Deep layer
                if i == 0:
                    self.layer1out = self.FM
                else:
                    self.layer2out = self.FM


            self.FM = tf.add(tf.matmul(self.FM, self.weights['right_layer_2']),
                             self.weights['right_bias_2'])  #
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                scope_bn='bn_%d' % i)  # None * layer[i] * 1
            self.FM = tf.add(self.FM, self.layer1out)
            self.FM = self.activation_function(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[2])


            self.FM = tf.add(tf.matmul(self.FM, self.weights['right_layer_3']),
                             self.weights['right_bias_3'])  #
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                scope_bn='bn_%d' % i)  # None * layer[i] * 1
            self.FM = tf.add(self.FM, self.layer2out)  #
            self.FM = self.activation_function(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[3])
            rightout = self.FM
            print(rightout.shape)

            self.finalout = tf.concat([leftout, rightout], 1)



            # Prediction Layer
            self.out = tf.matmul(self.finalout, self.weights['prediction']) + self.weights[
                'prediction_bias']

            self.prob = tf.sigmoid(self.out)

            # Compute the loss.
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(
                        tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                # self.out = tf.sigmoid(self.out)
                if self.lamda_bilinear > 0:
                    # self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, epsilon=1e-07,
                    #                                        scope=None) + tf.contrib.layers.l2_regularizer(
                    #     self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels) \
                        + tf.contrib.layers.l2_regularizer(
                            self.lamda_bilinear)(self.weights['feature_embeddings']))  # regulizer

                    # self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels) \
                    #             + tf.contrib.layers.l2_regularizer(
                    #     self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    # self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, epsilon=1e-07, scope=None)
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels)) \

                    # self.loss = tf.reduce_mean(
                    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels))

            # Optimizer.
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'ftrl':
                self.optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'adaldeta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            elif self.optimizer_type == 'padagrad':
                self.optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'pgd':
                self.optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss)
            elif self.optimizer_type == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

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
        if self.pretrain_flag > 0:  # with pretrain
            print('reading embeding from pretrain file...')
            pretrain_file = '../../output/pnn/pretrain/%s_10' % (
                args.dataset)
            weight_saver = tf.train.import_meta_graph(pretrain_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_linear_weights = pretrain_graph.get_tensor_by_name('linear_weights:0')
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, pretrain_file)
                fe, fb, b, l = sess.run([feature_embeddings, feature_bias, bias, feature_linear_weights])
            all_weights['linear_weights'] = tf.Variable(l, dtype=tf.float32)
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32)
            print('linear_weights:', all_weights['linear_weights'])
            print('feature_embeddings:', all_weights['feature_embeddings'])

        else:  # without pretrain
            all_weights['linear_weights'] = tf.Variable(
                tf.random_normal([self.features_M, 1], 0.0, 0.001),
                name='linear_weights')  # features_M * K

            maxval = np.sqrt(6. / np.sum([self.nn_input_dim, self.layers_1[0], self.layers_2[0]]))
            minval = -maxval
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_uniform([self.features_M, self.hidden_factor[1]], minval=minval,
                                  maxval=maxval, dtype=tf.float32), name='feature_embeddings', dtype=tf.float32)
            all_weights['bias_0'] = tf.Variable(
                tf.random_uniform([1, self.layers_1[0], self.layers_2[0]], minval=minval,
                                  maxval=maxval, dtype=tf.float32),
                dtype=tf.float32)  # 1 * layers[0]

            all_weights['feature_bias'] = tf.Variable(tf.random_uniform([self.features_M, 1], 0.0, 0.0),
                                                      name='feature_bias')  # features_M * 1
            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

        # attention
        if self.attention:
            glorot = np.sqrt(2.0 / (self.hidden_factor[0] + self.hidden_factor[1]))
            all_weights['attention_W'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor[1], self.hidden_factor[0])),
                dtype=np.float32, name="attention_W")  # K * AK
            all_weights['attention_b'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_factor[0])), dtype=np.float32,
                name="attention_b")  # 1 * AK
            all_weights['attention_p'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(self.hidden_factor[0])), dtype=np.float32,
                name="attention_p")  # AK

        # resnet_left
        letf_num_layer = len(self.layers_1)
        if letf_num_layer > 0:

            maxval = np.sqrt(6. / np.sum([self.nn_input_dim, self.layers_1[0]]))
            minval = -maxval
            all_weights['left_layer_0'] = tf.Variable(
                tf.random_uniform([self.nn_input_dim, self.layers_1[0]], minval=minval,
                                  maxval=maxval, dtype=tf.float32), dtype=tf.float32)
            all_weights['left_bias_0'] = tf.Variable(tf.random_uniform([1, self.layers_1[0]], minval=minval,
                                                                       maxval=maxval, dtype=tf.float32),
                                                     dtype=tf.float32)
            for i in range(1, letf_num_layer):
                glorot = np.sqrt(2.0 / (self.layers_1[i - 1] + self.layers_1[i]))
                all_weights['left_layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers_1[i - 1], self.layers_1[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['left_bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers_1[i])), dtype=np.float32)  # 1 * layer[i]
            # prediction layer

        # resnet_right
        right_num_layer = len(self.layers_2)
        if right_num_layer > 0:
            maxval = np.sqrt(6. / np.sum([self.nn_input_dim, self.layers_2[0]]))
            minval = -maxval
            all_weights['right_layer_0'] = tf.Variable(
                tf.random_uniform([self.nn_input_dim, self.layers_2[0]], minval=minval,
                                  maxval=maxval, dtype=tf.float32), dtype=tf.float32)
            # all_weights['bias_0'] = tf.Variable(tf.zeros([1, self.layers[0]], dtype=tf.float32), dtype=tf.float32)
            all_weights['right_bias_0'] = tf.Variable(tf.random_uniform([1, self.layers_2[0]], minval=minval,
                                                                        maxval=maxval, dtype=tf.float32),
                                                      dtype=tf.float32)

            for i in range(1, right_num_layer):
                glorot = np.sqrt(2.0 / (self.layers_2[i - 1] + self.layers_2[i]))
                all_weights['right_layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers_2[i - 1], self.layers_2[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['right_bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers_2[i])), dtype=np.float32)  # 1 * layer[i]

            # glorot = np.sqrt(6.0 / (self.layers_1[-1]+self.layers_2[-1] + 1))
            # all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=((self.layers_1[-1]+self.layers_2[-1]), 1)),
            #                                         dtype=np.float32)  # layers[-1] * 1
            maxval = np.sqrt(6.0 / (self.layers_1[-1] + self.layers_2[-1]))
            minval = -maxval
            # all_weights['prediction'] = tf.Variable(
            #     tf.random_uniform([(self.layers_1[-1] + self.layers_2[-1]), 1], minval=minval,
            #                       maxval=maxval, dtype=tf.float32), dtype=tf.float32)

            all_weights['prediction'] = tf.Variable(
                tf.random_uniform([self.layers_1[-1] + self.layers_2[-1], 1], minval=minval,
                                  maxval=maxval, dtype=tf.float32), dtype=tf.float32)

            all_weights['prediction_bias'] = tf.Variable(
                tf.random_uniform([1, 1], minval=minval,
                                  maxval=maxval, dtype=tf.float32), dtype=tf.float32)
        else:
            all_weights['prediction'] = tf.Variable(
                np.ones((self.hidden_factor[1], 1), dtype=np.float32))  # hidden_factor * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep_prob,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
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

    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, Train_data, Test_data):  # fit a dataset
        # Check Init performance
        # if self.verbose > 0:
        #     t2 = time()
        #     init_train, train_auc = self.evaluate(Train_data)
        #     init_valid, valid_auc = self.evaluate(Test_data)
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
                # batch_xs = self.get_next_block_from_data(Train_data, self.batch_size * i, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            train_result, train_auc = self.evaluate(Train_data)
            valid_result, valid_auc = self.evaluate(Test_data)

            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)

            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain_loss=%.8f\tvalid_loss=%.8f\ttrain_auc=%.8f\tvalid_auc=%.8f\t[%.1f s]"
                      % (epoch + 1, t2 - t1, train_result, valid_result, train_auc, valid_auc, time() - t2))

            if self.early_stop > 0 and self.eva_termination(self.valid_rmse):
                # print "Early stop at %d based on validation result." %(epoch+1)
                break

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

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        size = 30000
        total_batch = int(num_example / size)
        preds = []
        for i in range(total_batch + 1):
            batch_xs = self.get_next_block_from_data(data, size * i, size)
            feed_dict = {self.train_features: batch_xs['X'], self.train_labels: [y for y in batch_xs['Y']],
                         self.dropout_keep: self.no_dropout, self.train_phase: False}
            predictions = self.sess.run((self.prob), feed_dict=feed_dict)
            preds.extend(list(predictions))
        # print(preds[:5])
        y_pred = np.reshape(preds, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        auc = roc_auc_score(y_true, y_pred)
        if self.loss_type == 'square_loss':
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded,
                                             np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE, auc
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred)
            return logloss, auc


'''         # for testing the classification accuracy  
            predictions_binary = [] 
            for item in y_pred:
                if item > 0.5:
                    predictions_binary.append(1.0)
                else:
                    predictions_binary.append(0.0)
            Accuracy = accuracy_score(y_true, predictions_binary)
            return Accuracy '''

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)

    if args.verbose > 0:
        print(
            "AT_RES: dataset=%s, hidden_factor=%s, dropout_keep=%s, layers_1=%s,layers_2 = %s, loss_type=%s, pretrain=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d"
            % (args.dataset, args.hidden_factor, args.keep_prob, args.layers_1, args.layers_2, args.loss_type,
               args.pretrain, args.epoch,
               args.batch_size, args.lr, args.lamda, args.optimizer, args.batch_norm, args.activation, args.early_stop))
    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.nn.sigmoid
    elif args.activation == 'tanh':
        activation_function = tf.nn.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity
    elif activation_function == 'elu':
        activation_function = tf.nn.elu

    # Training
    t1 = time()
    model = NeuralFM(data.field_cnt, data.features_M, eval(args.hidden_factor), args.valid_dimen, args.attention, args.lamda_attention, eval(args.layers_1), eval(args.layers_2),
                     args.loss_type,
                     args.pretrain, args.epoch,
                     args.batch_size, args.lr, args.lamda, eval(args.keep_prob), args.optimizer, args.batch_norm,
                     activation_function, args.verbose, args.early_stop)
    model.train(data.Train_data, data.Test_data)

    # model.predict(data.Test_data)

    # Find the best validation result across iterations
    best_valid_score = 0
    if args.loss_type == 'square_loss':
        best_valid_score = min(model.valid_rmse)
    elif args.loss_type == 'log_loss':
        best_valid_score = min(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print("Best Iter(validation)= %d\t train_loss = %.8f, valid_loss = %.8f [%.1f s]"
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time() - t1))