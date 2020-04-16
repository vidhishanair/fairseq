# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from . import BaseWrapperDataset


class TruncateNDimDataset(BaseWrapperDataset):

    def __init__(self, dataset, truncation_length, dim=0):
        super().__init__(dataset)
        assert truncation_length is not None
        self.truncation_length = truncation_length
        self.dataset = dataset
        self.dim = dim

    def __getitem__(self, index):
        item = self.dataset[index]
        #print('pre truncate: '+str(item.size()))
        item_len = item.size(self.dim)
        item_len_sizes = [self.truncation_length if (idx == self.dim and item.size(idx) > self.truncation_length) else item.size(idx)
                          for idx in range(len(list(item.size())))]
        if item_len > self.truncation_length:
            item = item[:item_len_sizes[0], :item_len_sizes[1]]
        #print('post truncate: '+str(item.size()))
        return item

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)

    def __len__(self):
        return len(self.dataset)
