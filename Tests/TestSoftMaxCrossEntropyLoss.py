import unittest
import numpy as np
import Helpers

class TestSoftMaxCrossEntropyLoss(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = self.SoftMaxCrossEntropyLoss()
        loss = layer.forward(input_tensor, self.label_tensor)

        self.assertLess(loss, 1e-10)

    def test_backward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = self.SoftMaxCrossEntropyLoss()
        layer.forward(input_tensor, self.label_tensor)
        error = layer.backward(self.label_tensor)

        self.assertAlmostEqual(np.sum(error), 0)

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = self.SoftMaxCrossEntropyLoss()
        loss = layer.forward(input_tensor, self.label_tensor)

        # test a specific value here
        self.assertAlmostEqual(float(loss), 909.8875105980)

    def test_regression_backward_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = self.SoftMaxCrossEntropyLoss()
        layer.forward(input_tensor, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertGreaterEqual(element, 1 / 3)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertAlmostEqual(element, -1)

    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = self.SoftMaxCrossEntropyLoss()
        loss = layer.forward(input_tensor, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0.)

    def test_regression_backward(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = self.SoftMaxCrossEntropyLoss()
        layer.forward(input_tensor, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertGreaterEqual(element, 0)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertLessEqual(element, 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = self.SoftMaxCrossEntropyLoss()
        difference = Helpers.gradient_check([layer], input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_predict(self):
        input_tensor = np.arange(self.categories * self.batch_size)
        input_tensor = input_tensor / 100.
        input_tensor = input_tensor.reshape((self.batch_size, self.categories))
        layer = self.SoftMaxCrossEntropyLoss()
        prediction = layer.predict(input_tensor)
        expected_values = [[0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241],
                           [0.24626259, 0.24873757, 0.25123743, 0.25376241]]
        np.testing.assert_almost_equal(prediction, expected_values)

if __name__ == "__main__":
    pass