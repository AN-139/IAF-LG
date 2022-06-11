# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from Database import Database
from DISM import dism


class Fusion():
    def __init__(self, feedDict, embeddedX1):
        self.config = config
        self.X = tf.placeholder(tf.float32, [None, Config.seqLen, Config.model.embed_size])
        self.dism_X = tf.placeholder(tf.float32, [None, Config.seqLen, Config.model.dism_embed_size])
        self.pcm_X = tf.placeholder(tf.float32, [None, Config.seqLen, Config.model.pcm_embed_size])
        self.stylistic_vec = tf.placeholder(tf.float32, [None, Config.model.stylistic_size])
        self.Y = tf.placeholder(tf.float32, [None, Config.numClasses])
        self.weights = {
            'out': tf.Variable(tf.random_normal([2 * Config.bi_num_hidden, Config.numClasses])),
            'out_dism': tf.Variable(tf.random_normal([2 * Config.bi_num_hidden, Config.numClasses])),
            'out_pcm': tf.Variable(tf.random_normal([2 * Config.bi_num_hidden, Config.numClasses])),
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([Config.numClasses])),
            'out_dism': tf.Variable(tf.random_normal([Config.numClasses])),
            'out_pcm': tf.Variable(tf.random_normal([Config.numClasses])),
        }
        self.dropoutKeepProb = tf.placeholder(tf.float32)
        self.embeddedPosition = tf.placeholder(tf.float32, [None, config.seqLen, config.seqLen])
        self.loss = None
        self.bi_trans()
        pass

    def bi_trans(self):
        l2Loss = tf.constant(0.0)
        bi_x_outputs = tf.conpcmt(bi_x_outputs, 1)
        bi_x_outputs = tf.reshape(bi_x_outputs, [Config.batchSize, Config.seqLen, 2 * Config.bi_num_hidden])
        x_max_pooling = tf.reduce_max(bi_x_outputs, axis=2)
        bi_dism_x_outputs = BiLSTM(self.dism_X, self.weights['out_dism'], self.biases['out_dism'], name='dism_x')
        bi_dism_x_outputs = tf.conpcmt(bi_dism_x_outputs, 1)
        bi_dism_x_outputs = tf.reshape(bi_dism_x_outputs, [Config.batchSize, Config.seqLen, 2 * Config.bi_num_hidden])
        dism_x_max_pooling = tf.reduce_max(bi_dism_x_outputs, axis=2)
        gate_a_x_fuse = self.gate_A(x_max_pooling, dism_x_max_pooling)

        bi_pcm_x_outputs = BiLSTM(self.X, self.weights['out_pcm'], self.biases['out_pcm'], name='pcm_x') 
        bi_pcm_x_outputs = tf.conpcmt(bi_pcm_x_outputs, 1)
        bi_pcm_x_outputs = tf.reshape(bi_pcm_x_outputs, [Config.batchSize, Config.seqLen, 2 * Config.bi_num_hidden])
        pcm_x_max_pooling = tf.reduce_max(bi_pcm_x_outputs, axis=2)
        gate_a_pcm_fuse = self.gate_A(pcm_x_max_pooling, dism_x_max_pooling)

        fuse_x, fuse_dism_x, fuse_pcm_x = self.gate_C(gate_a_x_fuse, gate_a_pcm_fuse)

        feedDict_X = {'embeddedX': bi_x_outputs,
                      'dropoutKeepProb': self.dropoutKeepProb, 'embeddedPosition': self.embeddedPosition}
        feedDict_dism_X = {'embedded_dism_X': bi_dism_x_outputs,
                    'dropoutKeepProb': self.dropoutKeepProb, 'embeddedPosition': self.embeddedPosition}
        feedDict_pcm_X = {'embedded_pcm_X': bi_pcm_x_outputs,
                          'dropoutKeepProb': self.dropoutKeepProb, 'embeddedPosition': self.embeddedPosition}
        transformer = Transformer(self.config, feedDict_X, fuse_x)
        outputs = transformer.outputs
        transformer_dism = Transformer_DISM(self.config, feedDict_dism_X, fuse_dism_x)
        outputs_dism = transformer_dism.outputs
        transformer_pcm = Transformer_pcm(self.config, feedDict_pcm_X, fuse_pcm_x)
        outputs_pcm = transformer_pcm.outputs

        x_dism_x_out = self.gate_B(self.stylistic_vec, outputs, outputs_dism)
        outputs_con = self.gate_A(x_dism_x_out, pcm_x_pcm_out)
        outputSize = outputs_con.get_shape()[-1].value
        with tf.name_scope("output1"):
            outputW = tf.get_variable(
                "outputW1",
                shape=[outputSize, self.config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[self.config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.rumPredictions = tf.nn.xw_plus_b(outputs_con, outputW, outputB, name="predictions")
            predictResults = tf.argmax(self.rumPredictions, 1)
        with tf.name_scope("loss1"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.rumPredictions)
            self.loss = tf.reduce_mean(losses) + self.config.model.l2RegLambda * l2Loss
        pass


    def gate(self, vec1, vec2):
        sig_w = tf.Variable(tf.random_normal([2*Config.seqLen, 1]))
        sig_b = tf.Variable(tf.random_normal([1]))
        vec12 = tf.conpcmt([vec1, vec2], axis=1)
        sigm = tf.sigmoid(tf.nn.xw_plus_b(vec12, sig_w, sig_b))
        fusion = sigm*vec1 + (1-sigm)*vec2
        tan_x_w = tf.Variable(tf.random_normal([Config.seqLen, Config.seqLen]))
        tan_x_b = tf.Variable(tf.random_normal([Config.seqLen]))
        vec_x = tf.tanh(tf.nn.xw_plus_b(fusion, tan_x_w, tan_x_b))
        tan_dism_x_w = tf.Variable(tf.random_normal([Config.seqLen, Config.seqLen]))
        tan_dism_x_b = tf.Variable(tf.random_normal([Config.seqLen]))
        vec_dism_x = tf.tanh(tf.nn.xw_plus_b(fusion, tan_dism_x_w, tan_dism_x_b))
        tan_pcm_x_w = tf.Variable(tf.random_normal([Config.seqLen, Config.seqLen]))
        tan_pcm_x_b = tf.Variable(tf.random_normal([Config.seqLen]))
        vec_pcm_x = tf.tanh(tf.nn.xw_plus_b(fusion, tan_pcm_x_w, tan_pcm_x_b))
        vec_x = tf.expand_dims(vec_x, 1)
        vec_dism_x = tf.expand_dims(vec_dism_x, 1)
        vec_pcm_x = tf.expand_dims(vec_pcm_x, 1)
        return vec_x, vec_dism_x, vec_pcm_x

