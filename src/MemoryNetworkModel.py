import keras.backend as K
import numpy as np
from keras.engine.topology import Layer


class MemoryRepresentation(Layer):
    def __init__(self, output_dim, num_hops, **kwargs):
        self.output_dim = output_dim
        self.num_hops = num_hops
        super(MemoryRepresentation, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][2]
        initial_A_value = np.random.uniform(0, 1, size=[self.output_dim, input_dim])
        initial_B_value = np.random.uniform(0, 1, size=[self.output_dim, input_dim])
        # initial_H_value = np.random.uniform(0, 1, size=[self.output_dim, self.output_dim])
        initial_C_value = np.random.uniform(0, 1, size=[self.output_dim, input_dim])
        self.input_dim = input_shape[0][1]
        self.A = K.variable(initial_A_value)
        self.B = K.variable(initial_B_value)
        # self.H = K.variable(initial_H_value)
        self.C = K.variable(initial_C_value)
        self.trainable_weights = [self.A, self.B, self.C]
        super(MemoryRepresentation, self).build(input_shape)

    def call(self, inputs, mask=None):
        input_a = inputs[0]
        input_c = inputs[0]
        input_b = inputs[1]
        mem_m_tensor = K.dot(input_a, K.transpose(self.A))
        mem_c_tensor = K.dot(input_c, K.transpose(self.C))
        query_tensor = K.dot(input_b, K.transpose(self.B))

        softmax_tensor = K.softmax(K.reshape(K.dot(mem_m_tensor, K.transpose(query_tensor)), (1, self.input_dim)))
        output_tensor = K.reshape(K.dot(softmax_tensor, mem_c_tensor), (1, self.output_dim))
        response_tensor = output_tensor + query_tensor

        for _ in xrange(self.num_hops-1):
            softmax_tensor = K.softmax(K.reshape(K.dot(mem_m_tensor, K.transpose(response_tensor)), (1, self.input_dim)))
            output_tensor = K.reshape(K.dot(softmax_tensor, mem_c_tensor), (1, self.output_dim))
            response_tensor = output_tensor + response_tensor

        return response_tensor

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim


class OutputLayerW(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(OutputLayerW, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.random.uniform(0, 1, size=[input_dim, self.output_dim])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]
        super(OutputLayerW, self).build(input_shape)

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim
