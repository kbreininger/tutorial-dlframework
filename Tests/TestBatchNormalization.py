import unittest
import numpy as np
import Helpers

class TestBatchNorm(unittest.TestCase):

    def setUp(self):
        self.batch_size = 200
        self.channels = 2
        self.input_shape = (self.channels, 3, 3)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(0)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T

        self.categories = 5
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(self.FullyConnected(self.input_size, self.categories,0.))
        self.layers.append(Helpers.SoftMax())

    @staticmethod
    def _channel_moments(tensor, channels):
        in_shape = tensor.shape
        tensor = tensor.reshape(tensor.shape[0], channels, -1)
        tensor = np.transpose(tensor, (0, 2, 1))
        tensor = tensor.reshape(in_shape[1]//channels * in_shape[0], channels)
        mean = np.mean(tensor, axis=0)
        var = np.var(tensor, axis=0)
        return mean, var

    def test_forward_shape(self):
        layer = self.BatchNormalization(0.)
        output = layer.forward(self.input_tensor)

        self.assertEqual(output.shape[0], self.input_tensor.shape[0])
        self.assertEqual(output.shape[1], self.input_tensor.shape[1])

    def test_forward_shape_convolutional(self):
        layer = self.BatchNormalization(0., self.channels)
        output = layer.forward(self.input_tensor)

        self.assertEqual(output.shape[0], self.input_tensor.shape[0])
        self.assertEqual(output.shape[1], self.input_tensor.shape[1])

    def test_forward(self):
        layer = self.BatchNormalization(0.)
        output = layer.forward(self.input_tensor)
        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        self.assertAlmostEqual(np.sum(np.square(mean - np.zeros(mean.shape[0]))), 0)
        self.assertAlmostEqual(np.sum(np.square(var - np.ones(var.shape[0]))), 0)

    def test_forward_convolutional(self):
        layer = self.BatchNormalization(0., self.channels)
        output = layer.forward(self.input_tensor)
        mean, var = TestBatchNorm._channel_moments(output, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean)), 0)
        self.assertAlmostEqual(np.sum(np.square(var - np.ones_like(var))), 0)

    def test_forward_train_phase(self):
        layer = self.BatchNormalization(0.)
        layer.forward(self.input_tensor)

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertNotEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def test_forward_train_phase_convolutional(self):
        layer = self.BatchNormalization(0., self.channels)
        layer.forward(self.input_tensor)

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean, var = TestBatchNorm._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm._channel_moments(self.input_tensor, self.channels)

        self.assertNotEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def test_forward_test_phase(self):
        layer = self.BatchNormalization(0.)
        layer.forward(self.input_tensor)
        layer.phase = self.Phase.test

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertAlmostEqual(np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)
        self.assertAlmostEqual(np.sum(np.square(var)), 0)

    def test_forward_test_phase_convolutional(self):
        layer = self.BatchNormalization(0., self.channels)
        layer.forward(self.input_tensor)
        layer.phase = self.Phase.test

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean, var = TestBatchNorm._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm._channel_moments(self.input_tensor, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean + (mean_input / np.sqrt(var_input)))), 0, places=4)
        self.assertAlmostEqual(np.sum(np.square(var)), 0, places=4)

    def test_gradient(self):
        self.layers[0] = self.BatchNormalization(0.)

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_gradient_weights(self):
        self.layers[0] = self.BatchNormalization(0.)
        self.layers[0].forward(self.input_tensor)

        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, False)

        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias(self):
        self.layers[0] = self.BatchNormalization(0.)
        self.layers[0].forward(self.input_tensor)

        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, True)

        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_convolutional(self):
        self.layers[0] = self.BatchNormalization(0., self.channels)

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-3)

    def test_gradient_weights_convolutional(self):
        self.layers[0] = self.BatchNormalization(0., self.channels)
        self.layers[0].forward(self.input_tensor)

        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, False)

        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias_convolutional(self):
        self.layers[0] = self.BatchNormalization(0., self.channels)
        self.layers[0].forward(self.input_tensor)

        difference = Helpers.gradient_check_weights(self.layers, self.input_tensor, self.label_tensor, True)

        self.assertLessEqual(np.sum(difference), 1e-6)

if __name__ == "__main__":
    pass