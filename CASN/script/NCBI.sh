
FOLDER_PATH=/home/cp/CASN/
export PYTHONPATH=${FOLDER_PATH}

DATA_NAME="NCBI"
DATA_DIR="/home/cp/CASN/data/NCBI/"
BERT_DIR="/home/cp/Embedding/bert-base-cased/"
LABEL_DIR="/home/cp/CASN/data/NCBI/label_NCBI_768.npy"
MAXLEN=150
BATCH_SIZE=16
LR=2e-5
EPOCH=25
NER_DROPOUT=0.2
MAX_ENTITY_LEN=15
EARLY_STOP=5
CLIP_MAXNORM=1.0
Gamma=0.4

CUDA_VISIBLE_DEVICES=3 python -u LASN.py \
--data_name $DATA_NAME \
--data_dir $DATA_DIR \
--bert_model $BERT_DIR \
--max_seq_length $MAXLEN \
--batch_size $BATCH_SIZE \
--learning_rate $LR \
--num_train_epochs $EPOCH \
--gamma $Gamma \
--max_entity_span $MAX_ENTITY_LEN \
--early_stop $EARLY_STOP \
--clip_grad $CLIP_MAXNORM \
--ner_dropout $NER_DROPOUT \
--label_embed_url $LABEL_DIR