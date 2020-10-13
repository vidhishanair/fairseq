TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=1e-05
MAX_TOKENS=1024
UPDATE_FREQ=4
BART_PATH=bart.large/model.pt
SAVE_DIR=saved_models/subset_bart_mtokens1024_lr1e-5
#SAVE_DIR=saved_models/test2
mkdir $SAVE_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py /home/ubuntu/projects/datasets/cnn_dm_sentids-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --max-tokens-valid $MAX_TOKENS \
    --save-dir $SAVE_DIR \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters > $SAVE_DIR/train.log 2>&1 &
#--fp16
