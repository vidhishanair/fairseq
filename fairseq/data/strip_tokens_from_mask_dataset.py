# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from . import BaseWrapperDataset


class StripTokenFromMaskDataset(BaseWrapperDataset):

    def __init__(self, dataset, base_dataset, id_to_strip):
        super().__init__(dataset)
        self.id_to_strip = id_to_strip
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        base_item = self.base_dataset[index].ne(self.id_to_strip)
        #print('base_item : '+str(base_item.size()))
        #print('final item pre : '+str(item.size()))
        #print([item[i][base_item].unsqueeze(0).size() for i in range(item.size(0))])
        #print('final item : '+str(torch.cat([item[i][base_item].unsqueeze(0) for i in range(item.size(0))], dim=0).size()))
        return torch.cat([item[i][base_item].unsqueeze(0) for i in range(item.size(0))], dim=0)
