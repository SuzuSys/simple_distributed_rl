import unittest

import numpy as np
import srl
from srl.base.define import EnvObservationType
from srl.base.env.spaces.box import BoxSpace
from srl.envs import ox  # noqa F401
from srl.test import TestEnv
from srl.test.processor import TestProcessor


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test("OX")

    def test_player(self):
        self.tester.player_test("OX", "cpu")

    def test_processor(self):
        tester = TestProcessor()
        processor = ox.LayerProcessor()
        env_name = "OX"

        in_state = [0] * 9
        out_state = np.zeros((2, 3, 3))

        tester.run(processor, env_name)
        tester.change_observation_info(
            processor,
            env_name,
            EnvObservationType.SHAPE3,
            BoxSpace((2, 3, 3), 0, 1),
        )
        tester.observation_decode(
            processor,
            env_name,
            in_observation=in_state,
            out_observation=out_state,
        )

    def test_play_step(self):
        env = srl.make_env("OX")

        env.reset()
        np.testing.assert_array_equal(env.state, [0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(env.next_player_index == 0)

        # 1
        env.step(0)
        self.assertTrue(env.step_num == 1)
        self.assertTrue(env.next_player_index == 1)
        np.testing.assert_array_equal(env.state, [1, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(env.get_invalid_actions(), [0])
        np.testing.assert_array_equal(env.step_rewards, [0, 0])
        self.assertTrue(env.done == False)
        self.assertTrue(env.done_reason == "")

        # 2
        env.step(1)
        self.assertTrue(env.step_num == 2)
        self.assertTrue(env.next_player_index == 0)
        np.testing.assert_array_equal(env.state, [1, -1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1])
        np.testing.assert_array_equal(env.step_rewards, [0, 0])
        self.assertTrue(env.done == False)
        self.assertTrue(env.done_reason == "")

        # 3
        env.step(2)
        self.assertTrue(env.step_num == 3)
        self.assertTrue(env.next_player_index == 1)
        np.testing.assert_array_equal(env.state, [1, -1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2])
        np.testing.assert_array_equal(env.step_rewards, [0, 0])
        self.assertTrue(env.done == False)
        self.assertTrue(env.done_reason == "")

        # 4
        env.step(3)
        self.assertTrue(env.step_num == 4)
        self.assertTrue(env.next_player_index == 0)
        np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3])
        np.testing.assert_array_equal(env.step_rewards, [0, 0])
        self.assertTrue(env.done == False)
        self.assertTrue(env.done_reason == "")

        # 5
        env.step(4)
        self.assertTrue(env.step_num == 5)
        self.assertTrue(env.next_player_index == 1)
        np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(env.step_rewards, [0, 0])
        self.assertTrue(env.done == False)
        self.assertTrue(env.done_reason == "")

        # 6
        env.step(5)
        self.assertTrue(env.step_num == 6)
        self.assertTrue(env.next_player_index == 0)
        np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, -1, 0, 0, 0])
        np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(env.step_rewards, [0, 0])
        self.assertTrue(env.done == False)
        self.assertTrue(env.done_reason == "")

        # 7
        env.step(7)
        self.assertTrue(env.step_num == 7)
        self.assertTrue(env.next_player_index == 1)
        np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, -1, 0, 1, 0])
        np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4, 5, 7])
        np.testing.assert_array_equal(env.step_rewards, [0, 0])
        self.assertTrue(env.done == False)
        self.assertTrue(env.done_reason == "")

        # 8
        env.step(6)
        self.assertTrue(env.step_num == 8)
        self.assertTrue(env.next_player_index == 0)
        np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, -1, -1, 1, 0])
        np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(env.step_rewards, [0, 0])
        self.assertTrue(env.done == False)
        self.assertTrue(env.done_reason == "")

        # 9
        env.step(8)
        self.assertTrue(env.step_num == 9)
        self.assertTrue(env.next_player_index == 0)
        np.testing.assert_array_equal(env.state, [1, -1, 1, -1, 1, -1, -1, 1, 1])
        np.testing.assert_array_equal(env.get_invalid_actions(), [0, 1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(env.step_rewards, [1, -1])
        self.assertTrue(env.done == True)
        self.assertTrue(env.done_reason == "env")


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play_step", verbosity=2)
