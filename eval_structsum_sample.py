import files2rouge
#import
import torch
# from fairseq.models.bart import BARTModel
from fairseq.models.bart import StructSumBARTModel

#dirname = 'saved_models/subset10000_latent_str_mtokens1024_lr1e-5/'
#dirname = 'saved_models/cnn_latent_str_mtokens800_lr1e-5/'
import sys
fname = sys.argv[1]
dirname = 'saved_models/' + fname + '/' #init_test_999residual/'
bart = StructSumBARTModel.from_pretrained(
    dirname,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/home/ubuntu/projects/datasets_vid2/cnn_dm_sentids-bin'
)

bart.cuda()
bart.eval()
#bart.half()
count = 1
bsz = 16
with open('/home/ubuntu/projects/datasets_vid2/cnn_dm_sentids/test.source') as source,\
        open('/home/ubuntu/projects/datasets_vid2/cnn_dm_sentids/test.source-target.source.sentids') as source_sentids,\
        open(dirname+'/test.hypo', 'w') as fout:
    sline = source.readline().strip()
    sids = source_sentids.readline().strip()
    slines = [sline]
    sentids = [sids]
    for sline in source:
        sids = source_sentids.readline().strip()
        if count % 1000 == 0:
            print("Processed "+str(count)+" examples")
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []
            sentids = []

        slines.append(sline.strip())
        sentids.append(sids)
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()

hyp_path = dirname+"/test.hypo"
ref_path = '/home/ubuntu/projects/datasets_vid2/cnn_dm_sentids/test.target'
results = files2rouge.run(hyp_path, ref_path)
print(results)
wp = open(dirname+"/rouge_results.txt", 'w')
wp.write(str(results))
