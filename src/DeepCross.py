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
# from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


# tf.random_seed = 2017
# np.random_seed = 2017


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--path', nargs='?', default='../../output/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pnn',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=10,
                        help='Number of hidden factors.')
    parser.add_argument('--deep_layers', nargs='?', default='[500, 500, 500]',
                        help="Size of each layer.")
    parser.add_argument('--cross_layer_num', nargs='?', default='6',
                        help="number of cross layers.")
    parser.add_argument('--keep_prob', nargs='?', default='[.9, .9, .9, .9, .9]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0.00,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=.1,
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
    def __init__(self, field_cnt, features_M, hidden_factor, deep_layers, cross_layer_num, loss_type, pretrain_flag, epoch, batch_size,
                 learning_rate,
                 lamda_bilinear,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop, random_seed=2017):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.deep_layers = deep_layers
        self.cross_layer_num = cross_layer_num
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

        self.nn_input_dim = self.field_cnt * self.hidden_factor
        self.cross_dim = self.field_cnt*self.hidden_factor

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
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.

            # get the summed up embeddings of features.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)

            w_fm_nn_input = tf.reshape(nonzero_embeddings, [-1, self.field_cnt * self.hidden_factor])

            self.FM = w_fm_nn_input

            print('w_fm_nn_input.shape:', w_fm_nn_input.shape)

            # ________ Deep Layers __________
            for i in range(0, len(self.deep_layers)):
                self.FM = tf.add(tf.matmul(self.FM, self.weights['deep_layer_%d' % i]),
                                 self.weights['deep_bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                    scope_bn='deep_bn_%d' % i)  # None * layer[i] * 1
                self.FM = self.activation_function(self.FM)
                self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i])  # dropout at each Deep layer
            self.deep_net_out = self.FM

            # ________ product Layers __________
            x_0 = tf.reshape(nonzero_embeddings, [-1, self.field_cnt*self.hidden_factor, 1])  #定义x0
            x_1 = x_0
            for i in range(0, self.cross_layer_num):   #x0*xT*w+b+x
                dot = tf.matmul(x_0, x_1, transpose_b=True)
                dot = tf.tensordot(dot, self.weights['cross_layer_%d' % i], 1)
                x_1 = tf.add(dot, self.weights['cross_bias_%d' % i]) + x_1

            self.cross_net_out = tf.reshape(x_1, [-1, self.field_cnt*self.hidden_factor])

            x_stack = tf.concat([self.deep_net_out, self.cross_net_out], 1)

            self.out = tf.matmul(x_stack, self.weights['prediction'])  # None * 1
            # print(self.out)

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
                    #            + tf.contrib.layers.l2_regularizer(
                    #    self.lamda_bilinear)(self.weights['feature_embeddings']) # regulizer

                else:
                    # self.loss = tf.contrib.losses.log_loss(self.prob, self.train_labels, epsilon=1e-07, scope=None)
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels))

            # Optimizer.
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999,
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
            # elif self.optimizer_type == 'nadam':
            #     self.optimizer = tf.train.Na(learning_rate=self.learning_rate).minimize(self.loss)

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
            pretrain_file = '../../output/pnn/pretrain/%s_%d' % (
                args.dataset, args.hidden_factor)
            # pretrain_file = '../pretrain/%s_%d/%s_%d' % (
            #     args.dataset, args.hidden_factor, args.dataset, args.hidden_factor)
            weight_saver = tf.train.import_meta_graph(pretrain_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, pretrain_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else:  # without pretrain
            all_weights['linear_weights'] = tf.Variable(
                tf.random_normal([self.features_M, 1], 0.0, 0.001),
                name='linear_weights')  # features_M * K
            #
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.001),
                name='feature_embeddings')  # features_M * K
            all_weights['bias_0'] = tf.Variable(tf.random_normal([1, self.deep_layers[0]], 0.0, 0.001))

            # maxval = np.sqrt(6. / np.sum([self.nn_input_dim, self.deep_layers[0]]))
            # minval = -maxval
            # all_weights['feature_embeddings'] = tf.Variable(
            #     tf.random_uniform([self.features_M, self.hidden_factor], minval=minval,
            #                       maxval=maxval, dtype=tf.float32), name='feature_embeddings', dtype=tf.float32)
            # all_weights['bias_0'] = tf.Variable(tf.random_uniform([1, self.deep_layers[0]], minval=minval,
            #                                                       maxval=maxval, dtype=tf.float32),
            #                                     dtype=tf.float32)  # 1 * layers[0]

            # all_weights['feature_bias'] = tf.Variable(tf.random_uniform([self.features_M, 1], 0.0, 0.0),
            #                                           name='feature_bias')  # features_M * 1
            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1


        # deep layers
        deep_layer_num = len(self.deep_layers)
        if deep_layer_num > 0:
            maxval = np.sqrt(6. / np.sum([self.nn_input_dim, self.deep_layers[0]]))
            minval = -maxval
            all_weights['deep_layer_0'] = tf.Variable(
                tf.random_uniform([self.nn_input_dim, self.deep_layers[0]], minval=minval,
                                  maxval=maxval, dtype=tf.float32), dtype=tf.float32)
            all_weights['deep_bias_0'] = tf.Variable(tf.random_uniform([1, self.deep_layers[0]], minval=minval,
                                                                       maxval=maxval, dtype=tf.float32),
                                                     dtype=tf.float32)  # 1 * layers[0]

            for i in range(1, deep_layer_num):
                glorot = np.sqrt(6.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                all_weights['deep_layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['deep_bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])), dtype=np.float32)  # 1 * layer[i]

        if self.cross_layer_num > 0:
            maxval = np.sqrt(6. / np.sum([self.nn_input_dim, self.deep_layers[0]]))
            minval = -maxval
            all_weights['cross_layer_0'] = tf.Variable(
                tf.random_uniform([self.cross_dim, 1], minval=minval,
                                  maxval=maxval, dtype=tf.float32), dtype=tf.float32)
            all_weights['cross_bias_0'] = tf.Variable(tf.random_uniform([self.cross_dim, 1], minval=minval,
                                                                        maxval=maxval, dtype=tf.float32),
                                                      dtype=tf.float32)

            for i in range(1, self.cross_layer_num):
                glorot = np.sqrt(6.0 / (2 * self.cross_dim))

                all_weights['cross_layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.cross_dim, 1)),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['cross_bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.cross_dim, 1)),
                    dtype=np.float32)  # 1 * layer[i]

        if deep_layer_num > 0 and self.cross_layer_num > 0:
            # prediction layer
            glorot = np.sqrt(6.0 / (self.deep_layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.deep_layers[-1] + self.cross_dim, 1)),
                dtype=np.float32)  # layers[-1] * 1
        else:
            all_weights['prediction'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1
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

    def train(self, Train_data, Validation_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train, train_auc = self.evaluate(Train_data)
            init_valid, valid_auc = self.evaluate(Validation_data)

            print("Init: \t train_loss=%.8f\tvalid_loss=%.8f\ttrain_auc=%.8f\tvalid_auc=%.8f\t[%.1f s]" % (
                init_train, init_valid, train_auc, valid_auc, time() - t2))

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
            valid_result, valid_auc = self.evaluate(Validation_data)

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
                if valid[1] < valid[2] and valid[2] < valid[3] and valid[3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        size = 10000
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
            "Neural FM: dataset=%s, hidden_factor=%d, dropout_keep=%s, layers=%s, loss_type=%s, pretrain=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d"
            % (args.dataset, args.hidden_factor, args.keep_prob, args.deep_layers, args.loss_type, args.pretrain, args.epoch,
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
    model = NeuralFM(data.field_cnt, data.features_M, args.hidden_factor, eval(args.deep_layers), eval(args.cross_layer_num), args.loss_type,
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
    print("Best Iter(validation)= %d\t train = %.8f, valid = %.8f [%.1f s]"
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time() - t1))
#效果不错  3427效果
# Epoch 1 [713.5 s]	train_loss=0.00608091	valid_loss=0.00564546	train_auc=0.64897488	valid_auc=0.65649497	[317.3 s]
# Epoch 2 [691.9 s]	train_loss=0.00607109	valid_loss=0.00563914	train_auc=0.65864237	valid_auc=0.66569447	[304.8 s]
# Epoch 3 [677.2 s]	train_loss=0.00605729	valid_loss=0.00562374	train_auc=0.66917649	valid_auc=0.67532977	[301.7 s]
# Epoch 4 [676.9 s]	train_loss=0.00604223	valid_loss=0.00560753	train_auc=0.67101573	valid_auc=0.67732169	[301.1 s]
# Epoch 5 [678.2 s]	train_loss=0.00602555	valid_loss=0.00559668	train_auc=0.67934785	valid_auc=0.68744389	[301.5 s]
# Epoch 6 [676.9 s]	train_loss=0.00600192	valid_loss=0.00557352	train_auc=0.67803221	valid_auc=0.68540445	[301.4 s]
# Epoch 7 [677.3 s]	train_loss=0.00597737	valid_loss=0.00555085	train_auc=0.67866568	valid_auc=0.68644360	[301.6 s]
# Epoch 8 [676.1 s]	train_loss=0.00595276	valid_loss=0.00553207	train_auc=0.68500948	valid_auc=0.69250995	[301.4 s]
# Epoch 9 [677.2 s]	train_loss=0.00592644	valid_loss=0.00550365	train_auc=0.68905884	valid_auc=0.69675737	[301.6 s]
# Epoch 10 [676.0 s]	train_loss=0.00590263	valid_loss=0.00549225	train_auc=0.69486089	valid_auc=0.70183981	[300.6 s]
# Epoch 11 [677.3 s]	train_loss=0.00587371	valid_loss=0.00546432	train_auc=0.69958148	valid_auc=0.70576427	[300.7 s]
# Epoch 12 [676.5 s]	train_loss=0.00584920	valid_loss=0.00545461	train_auc=0.70634829	valid_auc=0.71250794	[302.0 s]
# Epoch 13 [676.7 s]	train_loss=0.00581564	valid_loss=0.00542550	train_auc=0.71065661	valid_auc=0.71402572	[301.7 s]
# Epoch 14 [678.4 s]	train_loss=0.00578559	valid_loss=0.00540120	train_auc=0.71682232	valid_auc=0.71716123	[302.3 s]
# Epoch 15 [676.7 s]	train_loss=0.00576175	valid_loss=0.00537432	train_auc=0.72074700	valid_auc=0.71900079	[302.0 s]
# Epoch 16 [678.6 s]	train_loss=0.00573728	valid_loss=0.00536510	train_auc=0.72633351	valid_auc=0.72200341	[302.3 s]
# Epoch 17 [676.6 s]	train_loss=0.00571859	valid_loss=0.00535225	train_auc=0.73194510	valid_auc=0.72372194	[301.6 s]
# Epoch 18 [678.0 s]	train_loss=0.00570229	valid_loss=0.00535152	train_auc=0.73774739	valid_auc=0.72717677	[301.8 s]
# Epoch 19 [676.9 s]	train_loss=0.00568715	valid_loss=0.00535218	train_auc=0.74635674	valid_auc=0.73348704	[301.5 s]
# Epoch 20 [678.8 s]	train_loss=0.00567347	valid_loss=0.00534867	train_auc=0.75145275	valid_auc=0.73672790	[302.4 s]
# Epoch 21 [678.0 s]	train_loss=0.00565872	valid_loss=0.00533127	train_auc=0.75422711	valid_auc=0.73622302	[303.1 s]
# Epoch 22 [676.9 s]	train_loss=0.00564450	valid_loss=0.00532628	train_auc=0.76145281	valid_auc=0.73906237	[301.7 s]
# Epoch 23 [676.8 s]	train_loss=0.00563303	valid_loss=0.00532757	train_auc=0.76390926	valid_auc=0.73875794	[301.7 s]
# Epoch 24 [677.1 s]	train_loss=0.00562239	valid_loss=0.00530864	train_auc=0.76883452	valid_auc=0.74131333	[301.9 s]
# Epoch 25 [677.8 s]	train_loss=0.00560693	valid_loss=0.00531926	train_auc=0.77447742	valid_auc=0.74272212	[301.9 s]
# Epoch 26 [676.5 s]	train_loss=0.00560002	valid_loss=0.00529777	train_auc=0.77980740	valid_auc=0.74565428	[299.2 s]
# Epoch 27 [674.0 s]	train_loss=0.00557955	valid_loss=0.00530376	train_auc=0.78542375	valid_auc=0.74730741	[299.3 s]
# Epoch 28 [672.8 s]	train_loss=0.00557703	valid_loss=0.00534181	train_auc=0.78902680	valid_auc=0.74755821	[298.3 s]
# Epoch 29 [673.9 s]	train_loss=0.00556124	valid_loss=0.00533064	train_auc=0.79298054	valid_auc=0.75050362	[299.1 s]
# Epoch 30 [674.2 s]	train_loss=0.00555745	valid_loss=0.00528291	train_auc=0.79694586	valid_auc=0.75025896	[300.2 s]
# Epoch 31 [672.8 s]	train_loss=0.00553136	valid_loss=0.00528772	train_auc=0.80033280	valid_auc=0.74924845	[299.1 s]
# Epoch 32 [673.0 s]	train_loss=0.00551604	valid_loss=0.00529790	train_auc=0.80462779	valid_auc=0.75120282	[299.9 s]
# Epoch 33 [673.6 s]	train_loss=0.00550090	valid_loss=0.00528395	train_auc=0.80686852	valid_auc=0.75102569	[299.4 s]
# Epoch 34 [674.4 s]	train_loss=0.00551269	valid_loss=0.00537507	train_auc=0.80894512	valid_auc=0.75063810	[299.0 s]
# Epoch 35 [673.5 s]	train_loss=0.00547470	valid_loss=0.00531467	train_auc=0.81378288	valid_auc=0.75136841	[299.4 s]
# Epoch 36 [674.9 s]	train_loss=0.00545951	valid_loss=0.00532269	train_auc=0.81712750	valid_auc=0.75269015	[299.5 s]
# Epoch 37 [673.2 s]	train_loss=0.00543761	valid_loss=0.00528947	train_auc=0.82197157	valid_auc=0.75406892	[299.3 s]
# Epoch 38 [675.6 s]	train_loss=0.00544544	valid_loss=0.00536195	train_auc=0.82442013	valid_auc=0.75211400	[299.8 s]
# Epoch 39 [674.3 s]	train_loss=0.00541961	valid_loss=0.00533903	train_auc=0.82925873	valid_auc=0.75444381	[300.2 s]
# Epoch 40 [674.2 s]	train_loss=0.00539027	valid_loss=0.00527720	train_auc=0.83389648	valid_auc=0.75604437	[299.7 s]
# Epoch 41 [675.0 s]	train_loss=0.00537464	valid_loss=0.00526768	train_auc=0.83815383	valid_auc=0.75707055	[300.4 s]
# Epoch 42 [674.4 s]	train_loss=0.00535554	valid_loss=0.00526561	train_auc=0.84288345	valid_auc=0.75826318	[299.5 s]
# Epoch 43 [675.2 s]	train_loss=0.00532630	valid_loss=0.00529571	train_auc=0.84799667	valid_auc=0.75362695	[301.9 s]
# Epoch 44 [677.0 s]	train_loss=0.00530139	valid_loss=0.00527347	train_auc=0.85296374	valid_auc=0.75678463	[300.8 s]
# Epoch 45 [677.2 s]	train_loss=0.00527295	valid_loss=0.00531286	train_auc=0.86080020	valid_auc=0.75481124	[300.9 s]
# Epoch 46 [675.8 s]	train_loss=0.00524222	valid_loss=0.00527765	train_auc=0.86471166	valid_auc=0.75796333	[300.8 s]
# Epoch 47 [685.9 s]	train_loss=0.00519456	valid_loss=0.00530805	train_auc=0.87475663	valid_auc=0.75464365	[340.3 s]

# 3358效果
# Epoch 1 [438.1 s]	train_loss=0.00629441	valid_loss=0.00686708	train_auc=0.67199964	valid_auc=0.73478409	[190.1 s]
# Epoch 2 [437.6 s]	train_loss=0.00624435	valid_loss=0.00679036	train_auc=0.67697692	valid_auc=0.73066848	[189.9 s]
# Epoch 3 [437.6 s]	train_loss=0.00617633	valid_loss=0.00668838	train_auc=0.68134140	valid_auc=0.73378134	[189.5 s]
# Epoch 4 [437.2 s]	train_loss=0.00611394	valid_loss=0.00658922	train_auc=0.69974909	valid_auc=0.75353167	[189.4 s]
# Epoch 5 [436.9 s]	train_loss=0.00605718	valid_loss=0.00649794	train_auc=0.70800115	valid_auc=0.75877649	[189.4 s]
# Epoch 6 [437.6 s]	train_loss=0.00599552	valid_loss=0.00638263	train_auc=0.71495343	valid_auc=0.75978644	[189.6 s]
# Epoch 7 [437.0 s]	train_loss=0.00596144	valid_loss=0.00633047	train_auc=0.72526649	valid_auc=0.76699834	[189.5 s]
# Epoch 8 [438.1 s]	train_loss=0.00593489	valid_loss=0.00628414	train_auc=0.73338743	valid_auc=0.77570853	[189.6 s]
# Epoch 9 [437.9 s]	train_loss=0.00591305	valid_loss=0.00624788	train_auc=0.74401873	valid_auc=0.78115082	[189.7 s]
# Epoch 10 [438.2 s]	train_loss=0.00589345	valid_loss=0.00621843	train_auc=0.75211180	valid_auc=0.78531683	[189.7 s]
# Epoch 11 [438.1 s]	train_loss=0.00587740	valid_loss=0.00620863	train_auc=0.76342374	valid_auc=0.78938207	[189.7 s]
# Epoch 12 [438.9 s]	train_loss=0.00585253	valid_loss=0.00618162	train_auc=0.77014797	valid_auc=0.79522652	[189.9 s]
# Epoch 13 [438.5 s]	train_loss=0.00583395	valid_loss=0.00616473	train_auc=0.77661873	valid_auc=0.79813763	[189.9 s]
# Epoch 14 [438.8 s]	train_loss=0.00581840	valid_loss=0.00616049	train_auc=0.78302530	valid_auc=0.80020537	[189.7 s]
# Epoch 15 [438.8 s]	train_loss=0.00579538	valid_loss=0.00614173	train_auc=0.78654417	valid_auc=0.79884265	[189.9 s]
# Epoch 16 [439.4 s]	train_loss=0.00577608	valid_loss=0.00612833	train_auc=0.79293194	valid_auc=0.80260788	[190.1 s]
# Epoch 17 [438.8 s]	train_loss=0.00575919	valid_loss=0.00612221	train_auc=0.79533870	valid_auc=0.80246829	[190.0 s]
# Epoch 18 [438.5 s]	train_loss=0.00575022	valid_loss=0.00612497	train_auc=0.79996320	valid_auc=0.80411782	[190.0 s]
# Epoch 19 [438.7 s]	train_loss=0.00572988	valid_loss=0.00611184	train_auc=0.80524773	valid_auc=0.80463403	[190.1 s]
# Epoch 20 [439.1 s]	train_loss=0.00571360	valid_loss=0.00610184	train_auc=0.80833375	valid_auc=0.80447516	[190.0 s]
# Epoch 21 [439.8 s]	train_loss=0.00570048	valid_loss=0.00610692	train_auc=0.80901647	valid_auc=0.80039206	[190.8 s]
# Epoch 22 [441.0 s]	train_loss=0.00570735	valid_loss=0.00613567	train_auc=0.81094406	valid_auc=0.80061476	[191.8 s]
# Epoch 23 [441.0 s]	train_loss=0.00566574	valid_loss=0.00608552	train_auc=0.81740481	valid_auc=0.80392337	[190.9 s]
# Epoch 24 [441.5 s]	train_loss=0.00565914	valid_loss=0.00608450	train_auc=0.81977127	valid_auc=0.80568058	[190.8 s]
# Epoch 25 [441.2 s]	train_loss=0.00564588	valid_loss=0.00608190	train_auc=0.82468239	valid_auc=0.80678864	[191.0 s]
# Epoch 26 [440.5 s]	train_loss=0.00561856	valid_loss=0.00607529	train_auc=0.82841642	valid_auc=0.80712522	[191.3 s]
# Epoch 27 [441.5 s]	train_loss=0.00561292	valid_loss=0.00608753	train_auc=0.83107821	valid_auc=0.80526444	[190.6 s]
# Epoch 28 [441.6 s]	train_loss=0.00557989	valid_loss=0.00606460	train_auc=0.83466434	valid_auc=0.80589821	[191.4 s]
# Epoch 29 [441.7 s]	train_loss=0.00556976	valid_loss=0.00608104	train_auc=0.83874585	valid_auc=0.80799618	[191.1 s]
# Epoch 30 [441.9 s]	train_loss=0.00554121	valid_loss=0.00607703	train_auc=0.84041201	valid_auc=0.80593326	[191.2 s]
# Epoch 31 [441.5 s]	train_loss=0.00551779	valid_loss=0.00608153	train_auc=0.84690404	valid_auc=0.80806503	[191.1 s]
# Epoch 32 [441.7 s]	train_loss=0.00548849	valid_loss=0.00607480	train_auc=0.85110143	valid_auc=0.80758831	[190.8 s]
# Epoch 33 [442.0 s]	train_loss=0.00546284	valid_loss=0.00608307	train_auc=0.85579694	valid_auc=0.80918560	[190.8 s]
# Epoch 34 [441.7 s]	train_loss=0.00542759	valid_loss=0.00606569	train_auc=0.86097441	valid_auc=0.80972643	[190.9 s]
# Epoch 35 [442.2 s]	train_loss=0.00542112	valid_loss=0.00609756	train_auc=0.86818568	valid_auc=0.80965860	[190.9 s]
# Epoch 36 [441.5 s]	train_loss=0.00533759	valid_loss=0.00606069	train_auc=0.87509523	valid_auc=0.81083649	[191.0 s]
# Epoch 37 [445.4 s]	train_loss=0.00528112	valid_loss=0.00608034	train_auc=0.88084867	valid_auc=0.80935019	[191.0 s]
# Epoch 38 [442.0 s]	train_loss=0.00520833	valid_loss=0.00608661	train_auc=0.88842572	valid_auc=0.81161931	[191.4 s]
# Epoch 39 [442.7 s]	train_loss=0.00512281	valid_loss=0.00610607	train_auc=0.90285233	valid_auc=0.80755862	[191.1 s]
# Epoch 40 [442.5 s]	train_loss=0.00499481	valid_loss=0.00615045	train_auc=0.91119158	valid_auc=0.80431315	[191.1 s]
# Epoch 41 [442.9 s]	train_loss=0.00487499	valid_loss=0.00624445	train_auc=0.92176252	valid_auc=0.79776375	[191.3 s]
# Epoch 42 [442.6 s]	train_loss=0.00464155	valid_loss=0.00632722	train_auc=0.94210996	valid_auc=0.80902673	[191.4 s]
# Epoch 43 [443.1 s]	train_loss=0.00416968	valid_loss=0.00651540	train_auc=0.95794277	valid_auc=0.80117937	[191.6 s]
# Epoch 44 [442.8 s]	train_loss=0.00341231	valid_loss=0.00720413	train_auc=0.97916241	valid_auc=0.76448663	[191.6 s]
# Epoch 45 [443.3 s]	train_loss=0.00251860	valid_loss=0.00765937	train_auc=0.98952612	valid_auc=0.76418357	[191.7 s]
# Epoch 46 [443.0 s]	train_loss=0.00223204	valid_loss=0.00785649	train_auc=0.99469328	valid_auc=0.77933796	[191.6 s]