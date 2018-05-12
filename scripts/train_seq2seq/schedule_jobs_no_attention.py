import os
import pdb
import subprocess as sp

OUTPUT_ROOT='/scratch/cluster/pkar/text-style-transfer/cyclegan/tmp/seq2seq_1k_no_attention'
SCRIPT_ROOT='/scratch/cluster/pkar/text-style-transfer/cyclegan/scripts/train_seq2seq'

mapping_dict = { 
    # Condor Scheduling Parameters
    '__EMAILID__': 'pkar@cs.utexas.edu',
    '__PROJECT__': 'INSTRUCTIONAL', 
    # Algorithm hyperparameters
    '__CODE_ROOT__': '/scratch/cluster/pkar/text-style-transfer/cyclegan',
    '__BATCH_SIZE__': '128',
    '__DATA_FILE__': ['/scratch/cluster/pkar/text-style-transfer/cyclegan/data/yafc_formal.h5', '/scratch/cluster/pkar/text-style-transfer/cyclegan/data/bible_darby.h5', '/scratch/cluster/pkar/text-style-transfer/cyclegan/data/shakespeare_original.h5'],
    '__JSON_FILE__': ['/scratch/cluster/pkar/text-style-transfer/cyclegan/data/yafc_formal.json', '/scratch/cluster/pkar/text-style-transfer/cyclegan/data/bible_darby.json', '/scratch/cluster/pkar/text-style-transfer/cyclegan/data/shakespeare_original.json'],
    '__P_DATA_FILE__': ['/scratch/cluster/pkar/text-style-transfer/cyclegan/data/yafc_informal.h5', '/scratch/cluster/pkar/text-style-transfer/cyclegan/data/bible_ylt.h5', '/scratch/cluster/pkar/text-style-transfer/cyclegan/data/shakespeare_modern.h5'],
    '__P_DATA_JSON__': ['/scratch/cluster/pkar/text-style-transfer/cyclegan/data/yafc_informal.json', '/scratch/cluster/pkar/text-style-transfer/cyclegan/data/bible_ylt.json', '/scratch/cluster/pkar/text-style-transfer/cyclegan/data/shakespeare_modern.json'],
    '__SHUFFLE__': 'True',
    '__TRAIN_MODE__': 'train_seq2seq',
    '__EMBEDDING_SIZE__': '300',
    '__HIDDEN_SIZE__': '350',
    '__NUM_RNN_LAYERS__': '1',
    '__USE_LSTM__': 'True',
    '__USE_ATTENTION__': 'False',
    '__EPOCHS__': '100',
    '__LEARNING_RATE__': '1e-4',
    '__WEIGHT_DECAY__': '1e-4',
    '__DROPOUT_P__': '0.2',
    '__MAX_NORM__': '10',
    '__NUM_SAMPLE_SENTS__': '5',
    '__LOG_ITER__': '10',
    '__SENT_SAMPLE_ITER__': '100',
    '__MODEL_NAME__': 'model',
    '__PRETRAINED_LM1_MODEL_PATH__': ['/scratch/cluster/pkar/text-style-transfer/cyclegan/models/yafc_formal_lm_best.net', '/scratch/cluster/pkar/text-style-transfer/cyclegan/models/bible_darby_lm_best.net', '/scratch/cluster/pkar/text-style-transfer/cyclegan/models/shakespeare_original_lm_best.net'],
    '__PRETRAINED_LM2_MODEL_PATH__': ['/scratch/cluster/pkar/text-style-transfer/cyclegan/models/yafc_informal_lm_best.net', '/scratch/cluster/pkar/text-style-transfer/cyclegan/models/bible_ylt_lm_best.net', '/scratch/cluster/pkar/text-style-transfer/cyclegan/models/shakespeare_modern_lm_best.net'],
    '__PRETRAINED_SEQ2SEQ_MODEL_PATH__': ['/scratch/cluster/pkar/text-style-transfer/cyclegan/models/yafc_formal_informal.net', '/scratch/cluster/pkar/text-style-transfer/cyclegan/models/bible_darby_ylt.net', '/scratch/cluster/pkar/text-style-transfer/cyclegan/models/shakespeare_original_modern.net'],
    '__PRETRAINED_GLOVE_VECTOR_PATH__': '/scratch/cluster/pkar/text-style-transfer/cyclegan/data/glove/glove.twitter.27B.200d.txt',
    '__USE_GLOVE_EMBEDDINGS__': 'True',
    '__ENABLE_SCHEDULED_SAMPLING__': 'False',
    '__SCHEDULED_SAMPLING_DECAY_TYPE__': 'NA',
    '__SCHEDULED_SAMPLING_DECAY_FACTOR__': '0.0',
    '__SCHEDULED_SAMPLING_MIN_EPS__': '0.5',
    '__NUM_SEARCHES__': '1',
    '__G_UPDATE_STEP_DIFF__': '1',
    '__D_UPDATE_STEP_DIFF__': '3',
    '__LR_RATIO_D_BY_G__': '1.0',
    '__DISCOUNT_FACTOR__': '0.99',
    '__LAMDA_RL__': '1e-0',
    '__LAMDA_REC_II__': '1e-2',
    '__LAMDA_REC_IJ__': '1e-3',
    '__SKIP_WEIGHT_DECAY__': '0.995',
    '__FREEZE_EMBEDDINGS__': 'False',
    '__CLAMP_LOWER__': '-0.01',
    '__CLAMP_UPPER__': '0.01',
    '__D_PRETRAIN_NUM_EPOCHS__': '3',
    '__DISC_RECALIBRATE__': '100'
    }

# Figure out number of jobs to run
num_jobs = 1
for key, value in mapping_dict.iteritems():
    if type(value) == type([]):
        if num_jobs == 1:
            num_jobs = len(value)
        else:
            assert(num_jobs == len(value))

for idx in range(num_jobs):
    mapping_dict['__JOBNAME__'] = 'run_%.2d'%(idx)
    mapping_dict['__LOGNAME__'] = os.path.join(OUTPUT_ROOT, mapping_dict['__JOBNAME__']) 
    mapping_dict['__LOG_DIR__'] = mapping_dict['__LOGNAME__']
    mapping_dict['__SAVE_PATH__'] = mapping_dict['__LOGNAME__']
    sp.call('mkdir %s'%(mapping_dict['__LOGNAME__']), shell=True)
    condor_script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'condor_script.sh')
    script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'run_script.sh')
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'condor_script_proto.sh'), condor_script_path), shell=True)
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'run_proto.sh'), script_path), shell=True)
    for key, value in mapping_dict.iteritems():
        if type(value) == type([]):
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], condor_script_path), shell=True)
        else:
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, condor_script_path), shell=True)

    sp.call('condor_submit %s'%(condor_script_path), shell=True)
