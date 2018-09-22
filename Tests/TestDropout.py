import unittest
import numpy as np
import Helpers

class TestDropout(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10000
        self.input_size = 10

        self.input_tensor = np.ones((self.batch_size, self.input_size))

    def test_forward_trainTime(self):
        drop_layer = self.DropOut(0.5)
        output = drop_layer.forward(self.input_tensor)

        self.assertEqual(np.max(output), 2)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean, 1. * self.input_size, places=1)

    def test_forward_testTime(self):
        drop_layer = self.DropOut(0.5)
        drop_layer.phase = self.Phase.test
        output = drop_layer.forward(self.input_tensor)

        self.assertEqual(np.max(output), 1.)
        self.assertEqual(np.min(output), 1.)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertEqual(sum_over_mean, 1. * self.input_size)

    def test_backward(self):
        drop_layer = self.DropOut(0.5)
        drop_layer.forward(self.input_tensor)
        output = drop_layer.backward(self.input_tensor)

        self.assertEqual(np.max(output), 1)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean, .5 * self.input_size, places=1)

if __name__ == "__main__":
    pass