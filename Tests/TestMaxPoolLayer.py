import unittest
import numpy as np
import Helpers

class TestMaxPooling(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (2, 4, 7)

        np.random.seed(1337)
        self.input_tensor = np.abs(np.random.random((self.batch_size, *self.input_shape)))

        self.categories = 5
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(self.Flatten())
        self.layers.append(None)
        self.layers.append(Helpers.SoftMax())

    def test_shape(self):
        layer = self.MaxPooling(neighborhood=(2, 2), stride=(2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 3])
        self.assertEqual(np.abs(np.sum(np.array(result.shape) - expected_shape)), 0)

    def test_overlapping_shape(self):
        layer = self.MaxPooling(neighborhood=(2, 2), stride=(2, 1))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 6])
        self.assertEqual(np.abs(np.sum(np.array(result.shape) - expected_shape)), 0)

    def test_subsampling_shape(self):
        layer = self.MaxPooling(neighborhood=(2, 2), stride=(3, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 1, 3])
        self.assertEqual(np.abs(np.sum(np.array(result.shape) - expected_shape)), 0)

    def test_gradient_stride(self):
        self.layers[0] = self.MaxPooling(neighborhood=(2, 2), stride=(2, 2))
        self.layers[2] = self.FullyConnected(12, self.categories, 0.)

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_overlapping_stride(self):
        self.layers[0] = self.MaxPooling(neighborhood=(2, 2), stride=(2, 1))
        self.layers[2] = self.FullyConnected(24, self.categories, 0.)

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_subsampling_stride(self):

        self.layers[0] = self.MaxPooling(neighborhood=(2, 2), stride=(3, 2))
        self.layers[2] = self.FullyConnected(6, self.categories, 0.)

        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)

        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_layout_preservation(self):
        pool = self.MaxPooling(neighborhood=(1, 1), stride=(1, 1))
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = pool.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(output_tensor-input_tensor)), 0.)

    def test_expected_output_valid_edgecase(self):
        input_shape = (1, 3, 3)
        pool = self.MaxPooling(neighborhood=(2, 2), stride=(2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)

        result = pool.forward(input_tensor)
        expected_result = np.array([[4], [13]]).T
        self.assertEqual(np.abs(np.sum(result - expected_result)), 0)

    def test_expected_output(self):
        input_shape = (1, 4, 4)
        pool = self.MaxPooling(neighborhood=(2, 2), stride=(2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=np.float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)

        result = pool.forward(input_tensor)
        expected_result = np.array([[[[ 5.,  7.],[13., 15.]]],[[[21., 23.],[29., 31.]]]]).T
        self.assertEqual(np.abs(np.sum(result - expected_result)), 0)

if __name__ == "__main__":
    pass