# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class StripTokenFromMaskDataset(BaseWrapperDataset):

    def __init__(self, dataset, base_dataset, id_to_strip):
        super().__init__(dataset)
        self.id_to_strip = id_to_strip
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        base_item = self.base_dataset[index]
        return item[base_item.ne(self.id_to_strip).repeat(item.size(0), 1)]
