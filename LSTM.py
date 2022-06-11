import tensorflow as tf 
import numpy as np 
from LayerNormalizer import LayerNormalizer

class LSTM():
    def __init__(self, database, dim_input, dim_hidden, normalizeLayer = False):

        cnt = 0
        self.w_update = tf.Variable( initial_value = tf.random_normal_initializer()( shape = (dim_hidden, dim_input), dtype = weightDType ), trainable = True )
        cnt += 1
        self.u_update = tf.Variable( initial_value = tf.random_normal_initializer()( shape = (dim_hidden, dim_hidden), dtype = weightDType ), trainable = True )
        cnt += 1
        self.b_update = tf.Variable( initial_value = tf.zeros_initializer()( shape = (dim_hidden, 1), dtype = weightDType), trainable = True )
        cnt += 1
        
        self.w_reset = tf.Variable( initial_value = tf.random_normal_initializer()( shape = (dim_hidden, dim_input), dtype = weightDType ), trainable = True )
        cnt += 1
        self.u_reset = tf.Variable( initial_value = tf.random_normal_initializer()( shape = (dim_hidden, dim_hidden), dtype = weightDType ), trainable = True )
        cnt += 1
        self.b_reset = tf.Variable( initial_value = tf.zeros_initializer() ( shape = (dim_hidden, 1), dtype = weightDType ), trainable = True )
        cnt += 1

        self.w_memory = tf.Variable( initial_value = tf.random_normal_initializer()( shape = (dim_hidden, dim_input), dtype = weightDType ), trainable = True )
        cnt += 1
        self.u_memory = tf.Variable( initial_value = tf.random_normal_initializer()( shape = (dim_hidden, dim_hidden), dtype = weightDType ), trainable = True )
        cnt += 1

        if normalizeLayer:
            self.lnUpdate = LayerNormalizer(dim_batch = 1, dim_input = dim_hidden); cnt += self.lnUpdate.nWeightTensors
            self.lnReset = LayerNormalizer(dim_batch = 1, dim_input = dim_hidden); cnt += self.lnReset.nWeightTensors
            self.lnMemory = LayerNormalizer(dim_batch = 1, dim_input = dim_hidden); cnt += self.lnReset.nWeightTensors

        self.dim_hidden = dim_hidden
        self.dim_input = dim_input
        self.normalizeLayer = normalizeLayer

        self.nWeightTensors = cnt

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def __GetWeightsTensorList(self):
        list = []

        list.append(self.w_reset)
        list.append(self.u_reset)
        list.append(self.b_reset)
        list.append(self.w_update)
        list.append(self.u_update)
        list.append(self.b_update)
        list.append(self.w_memory)
        list.append(self.u_memory)
        
        if self.normalizeLayer:
            list = list + self.lnUpdate.weights
            list = list + self.lnReset.weights
            list = list + self.lnMemory.weights

        return list

    def __SetWeightsTensorList(self, list):
        cnt = 0
        self.w_reset = list[cnt]; cnt += 1
        self.u_reset = list[cnt]; cnt += 1
        self.b_reset = list[cnt]; cnt += 1
        self.w_update = list[cnt]; cnt += 1
        self.u_update = list[cnt]; cnt += 1
        self.b_update = list[cnt]; cnt += 1
        self.w_memory = list[cnt]; cnt += 1
        self.u_memory = list[cnt]; cnt += 1
        
        if self.normalizeLayer:
            self.lnUpdate.weights = list[ cnt: ]; cnt += self.lnUpdate.nWeightTensors
            self.lnReset.weights = list[ cnt: ]; cnt += self.lnReset.nWeightTensors
            self.lnMemory.weights = list[ cnt: ]; cnt += self.lnMemory.nWeightTensors
        
        self.nWeightTensors = cnt

    weights = property(__GetWeightsTensorList, __SetWeightsTensorList)

    def NWeights(self):
        sum = 0
        for tensor in self.weights:
            n = 1; k = 0
            for _ in range(len(tensor.shape)):
                n = n * tensor.shape[k]
            sum += n

        if self.normalizeLayer:
            sum += self.lnUpdate.NWeights()
            sum += self.lnReset.NWeights()
            sum += self.lnMemory.NWeights()

        return sum

    def GenerateStates(self, sequence):
        #assert isinstance(sequence, list)
        #assert len(sequence) > 0
        #for x in sequence:
        #    assert x.shape == [self.dim_input, 1]

        states = []; h_prev = tf.Variable( tf.zeros( shape = [self.dim_hidden, 1], dtype = weightDType ) )
        for x in sequence:
            hidden = self.GenerateSingleState(h_prev, x)
            #assert hidden.shape == [self.dim_hidden, 1]
            states.append(hidden)
        
        return states

    def GenerateSingleState(self, h_prev, x_curr):

        tf.debugging.assert_all_finite(x_curr, message = 'x_curr is a nan.')

        # Update gate
        #assert h_prev.shape == [self.dim_hidden, 1]
        #assert x_curr.shape == [self.dim_input, 1]
        #assert self.w_update.shape == [self.dim_hidden, self.dim_input]
        #assert self.u_update.shape == [self.dim_hidden, self.dim_hidden]
        a = tf.sigmoid(tf.matmul(self.w_update, x_curr), name = 'matmul - 1')
        b = tf.sigmoid(tf.matmul(self.u_update, h_prev), name = 'matmul - 2')
        if self.normalizeLayer:
            gate_update = tf.transpose( tf.sigmoid( self.lnUpdate.FeedForward( tf.transpose(tf.add(a, b)) ) ) )
        else:
            gate_update = tf.sigmoid( tf.add(a, b) )
        assert gate_update.shape == [self.dim_hidden, 1]

        # Reset gate
        #assert self.w_reset.shape == [self.dim_hidden, self.dim_input]
        #assert self.u_reset.shape == [self.dim_hidden, self.dim_hidden]
        a = tf.matmul(self.w_reset, x_curr, name = 'matmul - 3')
        b = tf.matmul(self.u_reset, h_prev, name = 'matmul - 4')
        if self.normalizeLayer:
            gate_reset = tf.transpose( tf.sigmoid( self.lnReset.FeedForward( tf.transpose(tf.add(a, b)) ) ) )
        else:
            gate_reset = tf.sigmoid( tf.add(a, b) )
        #assert gate_reset.shape == [self.dim_hidden, 1]

        # Current memory content
        #assert self.w_memory.shape == [self.dim_hidden, self.dim_input]
        #assert self.u_memory.shape == [self.dim_hidden, self.dim_hidden]
        #assert h_prev.shape == [self.dim_hidden, 1]
        #assert gate_reset.shape == [self.dim_hidden, 1]
        a = tf.matmul(self.w_memory, x_curr, name = 'natmul - 5')
        b = tf.matmul(self.u_memory, h_prev, name = 'matmul - 6')
        b = tf.multiply(gate_reset, b, name = 'multiply - 1')
        if self.normalizeLayer:
            h = tf.transpose( tf.tanh( self.lnMemory.FeedForward( tf.transpose(tf.add(a, b)) ) ) )
        else:
            h = tf.tanh( tf.add( a, b ) )
        #assert h.shape == [self.dim_hidden, 1]

        # Tradeoff
        #assert gate_update.shape == [self.dim_hidden, 1]
        #assert h_prev.shape == [self.dim_hidden, 1]
        a = tf.multiply( gate_update, h_prev, name = 'multiply - 2')
        b = tf.add( tf.Variable( tf.ones( shape = [self.dim_hidden, 1], dtype = weightDType ) ), - gate_update)
        b = tf.multiply( b, h, name = 'multiply - 3' )
        h_curr = tf.add( a, b )
        #assert h_curr.shape == [self.dim_hidden, 1]

        tf.debugging.assert_all_finite(h_curr, message = 'h_curr is a nan.')

        return h_curr

    # Unit test

