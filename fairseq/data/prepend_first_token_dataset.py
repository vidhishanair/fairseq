# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class PrependFirstTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset):
        super().__init__(dataset)
        self._sizes = np.array(dataset.sizes) + 1

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item = torch.cat([item.new([item[0]]), item])
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        n += 1
        return n

    def size(self, index):
        n = self.dataset.size(index)
        n += 1
        return n
