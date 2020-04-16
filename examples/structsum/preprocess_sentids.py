import torch
import numpy as np

for subset in ['train', 'val', 'test']:
    print('Processing '+subset)
    prefix = subset+'.source-target.'
    sent_id_dataset = []
    sent_sizes = []
    with open(prefix + 'source.sentids', 'r', encoding='utf-8') as f:
        print(prefix + 'source.sentids')
        count = 0
        for line in f:
            if count % 1000 == 0:
                print('Processed '+str(count)+' examples')
            data = line.strip('\n').split(" ")
            data = list(map(int, data))
            data.append(data[-1])
            no_sents = data[-1]+1
            no_words = len(data)
            data = torch.LongTensor(data)
            one_hot_data = np.zeros((no_sents, no_words))
            for id in range(no_sents):
                one_hot_data[id, data.eq(id)] = 1
            sent_id_dataset.append(torch.from_numpy(one_hot_data))
            sent_sizes.append(no_words)
    save_obj = {'sent_id_dataset': sent_id_dataset, 'sent_sizes': sent_sizes}
    save_filename = prefix + 'source.sentids.pt'
    print('Saving to : '+save_filename)
    torch.save(save_obj, save_filename)
