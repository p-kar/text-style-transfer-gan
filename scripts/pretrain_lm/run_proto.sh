source /scratch/cluster/pkar/pytorch_gpu_27/bin/activate
code_root=__CODE_ROOT__

python $code_root/main.py --batch_size __BATCH_SIZE__ \
        --dataFile __DATA_FILE__ \
        --jsonFile __JSON_FILE__ \
        --pdataFile __P_DATA_FILE__ \
        --pjsonFile __P_DATA_JSON__ \
        --shuffle __SHUFFLE__ --train_mode __TRAIN_MODE__ --embedding_size __EMBEDDING_SIZE__ \
        --hidden_size __HIDDEN_SIZE__ --num_rnn_layers __NUM_RNN_LAYERS__ --use_lstm __USE_LSTM__ \
        --epochs __EPOCHS__ --lr __LEARNING_RATE__ --weight_decay __WEIGHT_DECAY__ \
        --dropout_p __DROPOUT_P__ --max_norm __MAX_NORM__ \
        --log_dir __LOG_DIR__ \
        --num_sample_sents __NUM_SAMPLE_SENTS__ --save_path __SAVE_PATH__ --model_name __MODEL_NAME__\
        --pretrained_lm1_model_path __PRETRAINED_LM1_MODEL_PATH__ \
        --pretrained_lm2_model_path __PRETRAINED_LM2_MODEL_PATH__ \
        --pretrained_seq2seq_model_path __PRETRAINED_SEQ2SEQ_MODEL_PATH__ \
        --pretrained_glove_vector_path __PRETRAINED_GLOVE_VECTOR_PATH__ \
        --use_glove_embeddings __USE_GLOVE_EMBEDDINGS__ \
        --num_searches __NUM_SEARCHES__ --g_update_step_diff __G_UPDATE_STEP_DIFF__ --d_update_step_diff __D_UPDATE_STEP_DIFF__ \
        --lr_ratio_D_by_G __LR_RATIO_D_BY_G__ --discount_factor __DISCOUNT_FACTOR__ \
        --lamda_rl __LAMDA_RL__ --lamda_rec_ii __LAMDA_REC_II__ --lamda_rec_ij __LAMDA_REC_IJ__ \
        --skip_weight_decay __SKIP_WEIGHT_DECAY__ \
        --freeze_embeddings __FREEZE_EMBEDDINGS__ --clamp_lower __CLAMP_LOWER__ --clamp_upper __CLAMP_UPPER__ \
        --d_pretrain_num_epochs __D_PRETRAIN_NUM_EPOCHS__ \
        --log_iter __LOG_ITER__ --sent_sample_iter __SENT_SAMPLE_ITER__
