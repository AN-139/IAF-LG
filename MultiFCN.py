import tensorflow as tf 

from LayerNormalizer import LayerNormalizer

class MultiFCN():

    def __init__(self, dim_batch, dim_input, dim_outputs, activations, useLayerNormalizer = False):
        rank = len(dim_outputs)
        assert rank > 0
        assert len(activations) == rank

        self.dim_batch = dim_batch
        self.dim_input = dim_input
        self.dim_output = dim_outputs[-1]
        self.rank = rank
        self.useLayerNormalizer = useLayerNormalizer

        self.w = [None] * rank
        self.b = [None] * rank
        self.f = [None] * rank
        self.ln = [None] * rank

        dim_input = dim_input
        cnt = 0
        for layer in range(rank):
            self.w[layer] = tf.Variable( initial_value = tf.random_normal_initializer()(shape = (dim_outputs[layer], dim_input), dtype = configWeightDType), trainable = True )
            cnt += 1
            self.b[layer] = tf.Variable( initial_value = tf.zeros_initializer()(shape = (dim_outputs[layer], 1), dtype = configWeightDType), trainable = True )
            cnt += 1

            dim_input = dim_outputs[layer]

            if activations[layer] == 'tanh': self.f[layer] = tf.tanh
            elif activations[layer] == 'sigmoid': self.f[layer] = tf.sigmoid
            else: raise Exception('Unacceptable activation required.')

            if useLayerNormalizer:
                self.ln[layer] = LayerNormalizer(dim_batch, dim_outputs[layer])
                cnt += self.ln[layer].nWeightTensors

        self.nWeightTensors = cnt

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def __GetWeightsTensorList(self):
        list = []

        for layer in range(self.rank):
            list.append(self.w[layer])
            list.append(self.b[layer])
            if self.useLayerNormalizer:
                list = list + self.ln[layer].weights

        return list

    def __SetWeightsTensorList(self, list):
        cnt = 0
        for layer in range(self.rank):
            self.w[layer] = list[cnt]; cnt += 1
            self.b[layer] = list[cnt]; cnt += 1
            if self.useLayerNormalizer:
                self.ln[layer].weights = list[cnt:]
                cnt += self.ln[layer].nWeightTensors

        self.nWeightTensors = cnt

    weights = property(__GetWeightsTensorList, __SetWeightsTensorList)

    def NWeights(self):
        sum = 0
        for tensor in self.weights:
            n = 1; k = 0
            for _ in range(len(tensor.shape)):
                n = n * tensor.shape[k]
            sum += n

        return sum
    
    def FeedForward(self, x):

        for layer in range(self.rank):
            assert x.shape == [ self.dim_batch, self.w[layer].shape[1] ]
            a = tf.transpose( tf.matmul( self.w[layer], tf.transpose(x) ) )
            assert a.shape == [ self.dim_batch, self.w[layer].shape[0] ]
            assert self.b[layer].shape == [ self.w[layer].shape[0], 1 ]
            a = tf.add( a, tf.transpose(self.b[layer]) )
            assert a.shape == [ self.dim_batch, self.w[layer].shape[0] ]
            if self.ln[layer] != None:
                a = self.ln[layer].FeedForward(x = a)
            assert a.shape == [ self.dim_batch, self.w[layer].shape[0] ]
            x = self.f[layer](a)
            assert x.shape == [ self.dim_batch, self.w[layer].shape[0] ]

        return x

    def FeedForwardList(self, inputList):
        print( 'MultiFCN =============================================')

        hiddenList = [None] * len(inputList)

        for n in range(len(inputList)):
            hidden = self.FeedForward( inputList[n] )
            hiddenList[n] = hidden

        return hiddenList

"""
# Unit test

O = MultiFCN(dim_batch = 2, dim_input = 3, dim_outputs = [4, 3], activations = ['tanh', 'sigmoid'], useLayerNormalizer = True)

x = tf.Variable( [ [1, 2, 3], [4, 5, 6] ], dtype = weightDType )

y = O.FeedForward(x)

print(y)

"""