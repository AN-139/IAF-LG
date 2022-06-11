import tensorflow as tf 

class LayerNormalizer():

    def __init__(self, dim_batch, dim_input):
        cnt = 0
        self.gamma = tf.Variable( tf.ones_initializer()(shape = (dim_batch, 1), dtype = configWeightDType ), trainable = True)
        cnt += 1
        self.beta = tf.Variable( tf.zeros_initializer()(shape = (dim_batch, 1), dtype = configWeightDType ), trainable = True)
        cnt += 1

        self.dim_batch = dim_batch
        self.dim_input = dim_input

        self.nWeightTensors = cnt

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def __GetWeightsTensorList(self):
        list = []
        list.append(self.gamma)
        list.append(self.beta)

        return list

    def __SetWeightsTensorList(self, list):
        cnt = 0
        self.gamma = list[cnt]; cnt += 1
        self.beta = list[cnt]; cnt += 1
        
        self.nWeightTensors == cnt

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
        assert x.shape == [self.dim_batch, self.dim_input]

        mu = tf.reduce_mean(x, axis = 1, keepdims = True)
        assert mu.shape == [self.dim_batch, 1]
        std = tf.sqrt( tf.reduce_mean(tf.pow( x - mu, 2), axis = 1, keepdims = True) + 1e-8 )  #tf.math.reduce_std(x - mu, axis = 0, keepdims = True)
        assert std.shape == [self.dim_batch, 1]
        x_hat = tf.add( x, - mu) / std
        assert x_hat.shape == [self.dim_batch, self.dim_input]
        out = tf.multiply(self.gamma, x_hat) + self.beta
        assert out.shape == [self.dim_batch, self.dim_input]

        return out
