# text-style-transfer-gan
Style transfer in text using cycle-consistent WGANs

## Architecture

<img src="imgs/CycleGAN.png"/>

## Requirements
Environment and Imports
You can either use the environment.yml file to set up a virtual environment or install the required packages.      
      
Use the environment.yml file by running:     
conda env create -f environment.yml      

- Python 2.7
- PyTorch 0.3.1
- TensorboardX
- h5py
- [coco-caption](https://github.com/tylin/coco-caption)

## Usage

### Instruction for running

For training on YAFC dataset:

1. Pretrain a LM for both `formal` and `informal` styles:

```bash
$ python main.py --batch_size 20 \
        --dataFile data/yafc_formal.h5 \
        --jsonFile data/yafc_formal.json \
        --shuffle True --train_mode pretrain_lm --embedding_size 300 \
        --hidden_size 350 --num_rnn_layers 1 --use_lstm True \
        --epochs 100 --lr 1e-4 --weight_decay 1e-4 \
        --dropout_p 0.5 --max_norm 10 \
        --log_dir logs/pretrain_lm/yafc_formal \
        --num_sample_sents 5 \
        --save_path models/pretrain_lm/yafc_formal --model_name model
```

Similarily for `informal`,

```bash
$ python main.py --batch_size 20 \
        --dataFile data/yafc_informal.h5 \
        --jsonFile data/yafc_informal.json \
        --shuffle True --train_mode pretrain_lm --embedding_size 300 \
        --hidden_size 350 --num_rnn_layers 1 --use_lstm True \
        --epochs 100 --lr 1e-4 --weight_decay 1e-4 \
        --dropout_p 0.5 --max_norm 10 \
        --log_dir logs/pretrain_lm/yafc_informal \
        --num_sample_sents 5 \
        --save_path models/pretrain_lm/yafc_informal --model_name model
```


2. Pretrain Seq2Seq model using MLE training that converts s1 to s2 and s2 back to s1 (we load pretrained LM weights to initialize generators):

```bash
$ python main.py --batch_size 128 \
        --dataFile data/yafc_formal.h5 \
        --jsonFile data/yafc_formal.json \
        --pdataFile data/yafc_informal.h5 \
        --pjsonFile data/yafc_informal.json \
        --shuffle True --train_mode train_seq2seq --embedding_size 300 \
        --hidden_size 350 --num_rnn_layers 1 --use_lstm True \
        --epochs 100 --lr 1e-4 --weight_decay 1e-4 \
        --dropout_p 0.2 --max_norm 10 \
        --log_dir logs/train_seq2seq \
        --num_sample_sents 5 \
        --save_path models/train_seq2seq --model_name model\
        --pretrained_lm1_model_path models/pretrain_lm/yafc_formal/model_best.net \
        --pretrained_lm2_model_path models/pretrain_lm/yafc_informal/model_best.net \
        --skip_weight_decay 0.995 \
        --log_iter 10 --sent_sample_iter 100
```

3. Finally, train the Seq2Seq model in `finetune_cyclegan` mode:

```bash
$ python main.py --batch_size 128 \
        --dataFile data/yafc_formal.h5 \
        --jsonFile data/yafc_formal.json \
        --pdataFile data/yafc_informal.h5 \
        --pjsonFile data/yafc_informal.json \
        --shuffle True --train_mode finetune_cyclegan --embedding_size 300 \
        --hidden_size 350 --num_rnn_layers 1 --use_lstm True --use_attention True\
        --epochs 100 --lr 5e-6 --weight_decay 1e-4 \
        --dropout_p 0.2 --max_norm 1 \
        --log_dir logs/finetune_cyclegan/ \
        --num_sample_sents 5 --save_path models/finetune_cyclegan/ --model_name model\
        --pretrained_lm1_model_path models/pretrain_lm/yafc_formal/model_best.net \
        --pretrained_lm2_model_path models/pretrain_lm/yafc_informal/model_best.net \
        --pretrained_seq2seq_model_path models/train_seq2seq/model_best.net \
        --num_searches 1 --g_update_step_diff 25 --d_update_step_diff 1 \
        --lr_ratio_D_by_G 20.0 --discount_factor 0.99 \
        --lamda_rl 1e-0 --lamda_rec_ii 1e-2 --lamda_rec_ij 1e-3 \
        --lamda_cos_ij 1e-1 \
        --freeze_embeddings True --clamp_lower -0.01 --clamp_upper 0.01 \
        --d_pretrain_num_epochs 3  --disc_recalibrate 400\
        --g_update_step_diff_recalib 200 \
        --log_iter 10 --sent_sample_iter 100
```

## Evaluation

We evaluate our models on BLEU score with n ranging between 1 and 4:

~~~~
$ python eval.py --model_path models/finetune_cyclegan/model_best.net \
        --dataFile data/yafc_formal.h5 \
        --jsonFile data/yafc_formal.json \
        --pdataFile data/yafc_informal.h5 \
        --pjsonFile data/yafc_informal.json \
        --split val_and_test
~~~~
