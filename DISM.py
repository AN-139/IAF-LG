#coding=utf-8

import tensorflow as tf
import numpy as np

class dism(object):
    def __init__(self, feedDict, embeddedX):

        self.embeddedX = embeddedX
        self.dropoutKeepProb = feedDict['dropout_keeprob']
        self.embeddedPosition = feedDict['embed_pos']
        self.outputs = None

        l2Loss = tf.constant(0.0)

        with tf.name_scope("embedding"):

            self.embeddedWords = tf.concat([self.embeddedX, self.embeddedPosition], -1)

        with tf.name_scope("transformer"):
            for i in range(Config.model.numBlocks):
                with tf.name_scope("transformer".format(i + 1)):
                    multiHeadAtt = self._multiheadAttention(rawKeys=self.embeddedX, queries=self.embeddedWords,
                                                            keys=self.embeddedWords)

                    self.embeddedWords = self._feedForward(multiHeadAtt,
                                                           [Config.model.filters,
                                                            Config.model.embed_size + Config.model.seq_len])

            outputs = tf.reshape(self.embeddedWords,
                                 [-1, Config.model.seq_len * (Config.model.embed_size + Config.model.seq_len)])

        outputSize = outputs.get_shape()[-1].value

        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropoutKeepProb)

        self.outputs = outputs

    def _layerNormalization(self, inputs, scope="layerNorm"):
        epsilon = Config.model.epsilon

        inputsShape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]

        paramsShape = inputsShape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(paramsShape))

        gamma = tf.Variable(tf.ones(paramsShape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        outputs = gamma * normalized + beta
        return outputs

    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="multiheadAttention"):

        numHeads = Config.model.numHeads
        keepProp = Config.model.keepProp

        if numUnits is None:
            numUnits = queries.get_shape().as_list()[-1]

        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)

        similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

        keyMasks = tf.sign(tf.abs(tf.reduce_sum(rawKeys, axis=-1)))

        keyMasks = tf.tile(keyMasks, [numHeads, 1])

        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,
                                  scaledSimilary)
        if causality:
            diagVals = tf.ones_like(maskedSimilary[0, :, :])  # [queries_len, keys_len]
            tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()  # [queries_len, keys_len]
            masks = tf.tile(tf.expand_dims(tril, 0),
                            [tf.shape(maskedSimilary)[0], 1, 1])  # [batch_size * numHeads, queries_len, keys_len]

            paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(masks, 0), paddings,
                                      maskedSimilary)  # [batch_size * numHeads, queries_len, keys_len]

        weights = tf.nn.softmax(maskedSimilary)

        outputs = tf.matmul(weights, V_)

        outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=keepProp)

        outputs += queries
        outputs = self._layerNormalization(outputs)
        return outputs

    def _feedForward(self, inputs, filters, scope="multiheadAttention"):

        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        outputs = tf.layers.conv1d(**params)
        outputs += inputs
        outputs = self._layerNormalization(outputs)

        return outputs

    def _positionEmbedding(self, scope="positionEmbedding"):
        batchSize = Config.batch_size
        sequenceLen = Config.model.seq_len
        embeddingSize = Config.model.embed_size

        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        positionEmbedding = np.array([[pos / np.power(10000, (i - i % 2) / embeddingSize) for i in range(embeddingSize)]
                                      for pos in range(sequenceLen)])

        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)
        return positionEmbedded
