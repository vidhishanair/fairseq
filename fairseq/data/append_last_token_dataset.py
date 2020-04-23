# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class AppendLastTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, split='None'):
        super().__init__(dataset)
        self._sizes = np.array(dataset.sizes) + 1
        self.split = split

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item = torch.cat([item, item[:,-1].unsqueeze(1)], dim=1)
        #print(self.split+' src_sent_append: '+str(item.size()))
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
