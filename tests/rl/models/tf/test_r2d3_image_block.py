import unittest

import numpy as np

from srl.utils.common import is_package_installed

try:
    from tensorflow.keras import layers as kl

    from srl.rl.models.tf.r2d3_image_block import R2D3ImageBlock
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def test_call(self):
        block = R2D3ImageBlock()
        batch_size = 16

        x = np.ones((batch_size, 96, 72, 3))
        out_x = block(x)

        self.assertTrue(out_x.shape == (batch_size, 256))

    def test_call_lstm(self):
        block = R2D3ImageBlock()
        batch_size = 16

        x = np.ones((batch_size, 1, 96, 72, 3))
        out_x = kl.TimeDistributed(block)(x)

        self.assertTrue(out_x.shape == (batch_size, 1, 256))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call_lstm", verbosity=2)
