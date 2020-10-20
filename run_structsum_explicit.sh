TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=1028
#MAX_TOKENS=1024
UPDATE_FREQ=4
BART_PATH=bart.large/model.pt
#SAVE_DIR=saved_models/cnn_latent_str_mtokens800_lr1e-5
SAVE_DIR=saved_models/$1 #layernorm_test_proper_residual
mkdir $SAVE_DIR

#CUDA_VISIBLE_DEVICES=0 python train.py /home/ubuntu/projects/datasets_vid2/cnn_dm_sentids_chains-bin \
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py /home/ubuntu/projects/datasets_vid2/cnn_dm_sentids_chains-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --max-tokens-valid $MAX_TOKENS \
    --save-dir $SAVE_DIR \
    --task structsum \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch structsum_bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --explicit_str_att \
    --identity_init \
    --use_layer_norm \
    --find-unused-parameters > $SAVE_DIR/train.log 2>&1 &
    #--find-unused-parameters 
    #--use_structured_attention \
    #--encoder-normalize-before \
#--fp16
#--restore-file $BART_PATH \
#--restore-file checkpoint_last.pt \
