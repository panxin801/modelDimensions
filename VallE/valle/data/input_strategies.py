import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import (Tuple, Type)
from lhotse import CutSet
from lhotse.dataset.collation import collate_features
from lhotse.dataset.input_strategies import (
    ExecutorType,
    PrecomputedFeatures,
    _get_executor,
)
from lhotse.utils import fastcopy


class PromptedFeatures:
    def __init__(self, prompts, features):
        self.prompts = prompts
        self.features = features

    def to(self, device):
        return PromptedFeatures(self.prompts.to(device), self.features.to(device))

    def sum(self):
        return self.features.sum()

    @property
    def ndim(self):
        return self.features.ndim

    @property
    def data(self,):
        return (self.prompts, self.features)
