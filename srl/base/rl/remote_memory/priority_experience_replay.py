from typing import Any, List, Optional, Tuple

import numpy as np
from srl.base.rl.base import RLRemoteMemory
from srl.rl.memories.proportional_memory import ProportionalMemory
from srl.rl.memories.rankbase_memory import RankBaseMemory
from srl.rl.memories.rankbase_memory_linear import RankBaseMemoryLinear
from srl.rl.memories.replay_memory import ReplayMemory


class PriorityExperienceReplay(RLRemoteMemory):
    def __init__(self, *args):
        super().__init__(*args)

    def init(self, name: str, capacity: int, alpha: float, beta_initial: float, beta_steps: int):

        memories = [
            ReplayMemory,
            ProportionalMemory,
            RankBaseMemory,
            RankBaseMemoryLinear,
        ]
        names = [m.getName() for m in memories]
        if name not in names:
            raise ValueError("Unknown memory({}). Memories is [{}].".format(name, ",".join(names)))

        for m in memories:
            if m.getName() == name:
                self.memory = m(capacity, alpha, beta_initial, beta_steps)
                break

    def length(self) -> int:
        return len(self.memory)

    def call_restore(self, data: Any, **kwargs) -> None:
        self.memory.restore(data)

    def call_backup(self, **kwargs):
        return self.memory.backup()

    # ---------------------------

    def add(self, batch: Any, td_error: Optional[float] = None):
        self.memory.add(batch, td_error)

    def sample(self, step: int, batch_size: int) -> Tuple[list, list, list]:
        return self.memory.sample(batch_size, step)

    def update(self, indices: List[int], batchs: List[Any], td_errors: np.ndarray) -> None:
        self.memory.update(indices, batchs, td_errors)
