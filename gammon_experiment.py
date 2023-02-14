from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple, cast

import numpy as np

import srl
import srl.base.rl.worker
from backgammon import config as gammon_config
from backgammon.gammon_api import BackGammon
from srl import runner
from srl.algorithms import stochastic_muzero
from srl.base.define import EnvAction, EnvObservation, EnvObservationType, Info
from srl.base.env import EnvBase, registration
from srl.base.env.space import SpaceBase
from srl.base.env.spaces import BoxSpace, DiscreteSpace


@dataclass
class GammonEnv(EnvBase):

    env = BackGammon(seed=0, turn_zero=True)

    @property
    def action_space(self) -> SpaceBase:
        """ アクションの取りうる範囲を返します"""
        return DiscreteSpace(gammon_config.ACTION_SPACE)

    @property
    def observation_space(self) -> SpaceBase:
        """ 状態の取りうる範囲を返します(SpaceBaseは後述) """
        return BoxSpace(shape=(gammon_config.OBS_DEPTH, gammon_config.OBS_WIDTH, gammon_config.OBS_HEIGHT), low=0, high=1)

    @property
    def observation_type(self) -> EnvObservationType:
        """ 状態の種類を返します。
        EnvObservationType は列挙型で以下です。
        DISCRETE  : 離散
        CONTINUOUS: 連続
        GRAY_2ch  : グレー画像(2ch)
        GRAY_3ch  : グレー画像(3ch)
        COLOR     : カラー画像
        SHAPE2    : 2次元空間
        SHAPE3    : 3次元空間
        """
        return EnvObservationType.SHAPE3

    @property
    def max_episode_steps(self) -> int:
        """ 1エピソードの最大ステップ数 """
        return 500

    @property
    def player_num(self) -> int:
        """ プレイヤー人数 """
        return 2

    def reset(self) -> Tuple[EnvObservation, int, Info]:
        """ 1エピソードの最初に実行。(初期化処理を実装)
        Returns: (
            init_state        : 初期状態
            next_player_index : 最初のプレイヤーの番号
            info              : 任意の情報
        )
        """
        self.env.reset(turn_zero=True)
        observation: EnvObservation = self.env.get_observation()
        return observation, 0, {}

    def step(
        self,
        action: EnvAction,
        player_index: int,
    ) -> Tuple[EnvObservation, List[float], bool, int, Info]:
        """ actionとplayerを元に1step進める処理を実装

        Args:
            action (EnvAction): player_index の action
            player_index (int): stepで行動するプレイヤーのindex

        Returns: (
            next_state : 1step後の状態
            [
                player1 reward,
                player2 reward,
                ...
            ],
            done,
            next_player_index,
            info,
        )
        """
        done: bool = self.env.action(cast(int, action))
        next_player: int = self.env.turn
        rewards: List[float] = [0, 0]
        rewards[next_player] = cast(float, done)
        observation: EnvObservation = self.env.get_observation()
        
        return (observation, rewards, done, next_player, {})


    # backup/restore で現環境を復元できるように実装
    def backup(self) -> Any:
        return deepcopy(self.env)
    def restore(self, data: Any) -> None:
        self.env = data

    # ------------------------------------
    # 以下は option です。（必須ではない）
    # ------------------------------------
    def close(self) -> None:
        """ 終了処理を実装 """
        pass
    
    # 描画関係
    def render_terminal(self, **kwargs) -> None:
        print(self.env)

    def render_rgb_array(self, **kwargs) -> Optional[np.ndarray]:
        return None

    def get_invalid_actions(self, player_index: int) -> list:
        """ 無効なアクションがある場合は配列で返す """
        actions: Set = set(range(gammon_config.ACTION_SPACE))
        legal_actions: Set = set(self.env.get_legal_actions())
        invalid_actions: Set = actions - legal_actions
        return list(invalid_actions)
    
    def make_worker(self, name: str) -> Optional["srl.base.rl.worker.WorkerRun"]:
        """ 環境に特化したAIを返す """
        return None

    def set_seed(self, seed: Optional[int] = None) -> None:
        """ 乱数seed """
        

registration.register(
    id='GammonEnv',
    entry_point=__name__ + ":GammonEnv",
    kwargs={},
)

config = runner.Config(
    env_config=srl.EnvConfig("GammonEnv"),
    rl_config=stochastic_muzero.Config(
        num_simulations=20,
        batch_size=128,
        discount=1.0,
        # 学習率
        lr_init=0.001,
        lr_decay_rate=0.1,
        lr_decay_steps=100_000,
        # カテゴリ化する範囲
        v_min=-10,
        v_max=10,
        # train
        unroll_steps=3,
        codebook_size=32,
        # model
        dynamics_blocks=15,
    ),
)

# --- train
parameter, memory, history = runner.train(config, timeout=60)
print(type(history))
