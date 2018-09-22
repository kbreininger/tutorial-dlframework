import unittest
import numpy as np
import Helpers

class TestReLU(unittest.TestCase):

    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.ones([self.batch_size, self.input_size])
        self.input_tensor[0:self.half_batch_size,:] -= 2

        self.label_tensor = np.zeros([self.batch_size, self.input_size])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_forward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 1

        layer = self.ReLU()
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_backward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 2

        layer = self.ReLU()
        layer.forward(self.input_tensor)
        output_tensor = layer.backward(self.input_tensor*2)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = list()
        layers.append(self.ReLU())
        layers.append(Helpers.SoftMax())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

class TestSigmoid(unittest.TestCase):
    Sigmoid = None

    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.abs(np.random.random((self.input_size, self.batch_size))).T
        self.input_tensor *= 2.
        self.input_tensor -= 1.

        self.label_tensor = np.zeros([self.input_size, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_forward(self):
        expected_tensor = 0.5 * (1. + np.tanh(self.input_tensor / 2.))

        layer = self.Sigmoid()
        output_tensor = layer.forward(self.input_tensor)
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = self.Sigmoid()
        output_tensor = layer.forward(self.input_tensor*2)

        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1.)
        self.assertGreaterEqual(out_min, 0.)

    def test_gradient(self):
        layers = list()
        layers.append(self.Sigmoid())
        layers.append(Helpers.SoftMax())
        difference = Helpers.gradient_check(layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

if __name__ == "__main__":
    pass