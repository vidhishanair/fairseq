import files2rouge
#import
import torch
# from fairseq.models.bart import BARTModel
from fairseq.models.bart import StructSumBARTModel
from fairseq.tasks.structsum_task import ChainsDataset

dirname = 'saved_models/explicit_nonorm_999/'
bart = StructSumBARTModel.from_pretrained(
    dirname,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/home/ubuntu/projects/datasets_vid2/cnn_dm_sentids_chains-bin'
)

bart.cuda()
bart.eval()
#bart.half()
count = 1
bsz = 8

chains_dataset = ChainsDataset("/home/ubuntu/projects/datasets_vid2/cnn_dm_sentids_chains/test.source-target.source.chains")

with open('/home/ubuntu/projects/datasets_vid2/cnn_dm_sentids_chains/test.source') as source, \
        open('/home/ubuntu/projects/datasets_vid2/cnn_dm_sentids_chains/test.source-target.source.sentids') as source_sentids, \
        open(dirname+'/test.hypo', 'w') as fout:
    sline = source.readline().strip()
    sids = source_sentids.readline().strip()
    #schains = source_chains.readline().strip()
    slines = [sline]
    sentids = [sids]
    sent_es_chains = [chains_dataset[0]]
    for idx, sline in enumerate(source):
        sids = source_sentids.readline().strip()
        schains = chains_dataset[idx+1]
        if count % 1000 == 0:
            print("Processed "+str(count)+" examples")
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3, chains_dataset=sent_es_chains)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []
            sentids = []
            sent_es_chains = []

        slines.append(sline.strip())
        sentids.append(sids)
        sent_es_chains.append(schains)
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3, chains_dataset=sent_es_chains)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()

hyp_path = dirname+"/test.hypo"
ref_path = '/home/ubuntu/projects/datasets_vid2/cnn_dm_sentids_chains/test.target'
results = files2rouge.run(hyp_path, ref_path)
print(results)
wp = open(dirname+"/rouge_results.txt", 'w')
wp.write(str(results))
