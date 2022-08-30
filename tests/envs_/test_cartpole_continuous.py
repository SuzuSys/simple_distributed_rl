import unittest

from envs import cartpole_continuous  # noqa F401
from srl.test import TestEnv


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test("CartpoleContinuous-v0")


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)