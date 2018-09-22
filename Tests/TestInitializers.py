import unittest
import numpy as np
import Helpers
from scipy import stats

class TestInitializers(unittest.TestCase):
    
    class DummyLayer:
        def __init__(self, input_size, output_size):
            self.weights = np.random.random_sample((output_size, input_size))
        
        def initialize(self, initializer):
            self.weights = initializer.initialize(self.weights.shape)
    
    def setUp(self):
        self.batch_size = 9
        self.input_size = 200
        self.output_size = 50
        
    def _performInitialization(self, initializer):
        np.random.seed(1337)
        layer = TestInitializers.DummyLayer(self.input_size, self.output_size)
        weights_before_init = layer.weights.copy()
        layer.initialize(initializer)
        weights_after_init = layer.weights.copy()
        return weights_before_init, weights_after_init
        
    def test_const_shape(self):
        weights_before_init, weights_after_init = self._performInitialization(self.Const(0.1))
        
        self.assertEqual(weights_before_init.shape, weights_after_init.shape)
        self.assertFalse(np.allclose(weights_before_init, weights_after_init))

    def test_const_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(self.Const(0.1))
        self.assertTrue(np.allclose(weights_after_init, 0.1))

    def test_uniform_shape(self):
        weights_before_init, weights_after_init = self._performInitialization(self.Uniform())
        
        self.assertEqual(weights_before_init.shape, weights_after_init.shape)
        self.assertFalse(np.allclose(weights_before_init, weights_after_init))

    def test_uniform_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(self.Uniform())

        p_value = stats.kstest(weights_after_init.flat, 'uniform', args=(0, 1)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_he_shape(self):
        weights_before_init, weights_after_init = self._performInitialization(self.He())
        
        self.assertEqual(weights_before_init.shape, weights_after_init.shape)
        self.assertFalse(np.allclose(weights_before_init, weights_after_init))

    def test_he_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(self.He())

        scale = np.sqrt(2.) / np.sqrt(self.input_size)
        p_value = stats.kstest(weights_after_init.flat, 'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)

if __name__ == "__main__":
    pass