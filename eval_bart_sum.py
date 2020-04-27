import torch
from fairseq.models.bart import BARTModel
#from fairseq.models.bart import StructSumBARTModel
bart = BARTModel.from_pretrained(
    'saved_models/bart_mtokens800/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/home/ubuntu/projects/datasets/cnn_dm-bin/'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open('/home/ubuntu/projects/datasets/cnn_dm/test.source') as source, open('saved_models/bart_mtokens800/test.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
