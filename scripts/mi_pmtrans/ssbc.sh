#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/hilo/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

model='dino'

DATASETS=("cubc" "scarsc" "fgvcc")
TASKs=("A_L+A_U+B->A_U+B+C")

SAVE_DIR=/disk/work/hjwang/HiLo/logs/mi_pmtrans_mixed+sampler/
WEIGHTS_PATH=/disk/work/hjwang/HiLo/data/ssbc/cubc.json

for d in ${!DATASETS[@]}; do
    for t in ${!TASKs[@]}; do
        echo ${DATASETS[$d]}
        ${PYTHON} -m methods.PMTrans.mi_dis_pm \
                    --dataset_name ${DATASETS[$d]} \
                    --batch_size 150 \
                    --grad_from_block 11 \
                    --epochs 200 \
                    --num_workers 8 \
                    --sup_weight 0.35 \
                    --weight_decay 5e-5 \
                    --warmup_teacher_temp 0.07 \
                    --teacher_temp 0.04 \
                    --warmup_teacher_temp_epochs 30 \
                    --memax_weight 2 \
                    --transform 'imagenet' \
                    --weights_path ${WEIGHTS_PATH} \
                    --lr 0.05 \
                    --eval_funcs 'v2' \
                    --src_env 'N/A' \
                    --tgt_env 'N/A' \
                    --aux_env 'N/A' \
                    --task_type ${TASKs[$t]} \
                    --model ${model} \
                    --model_path ${SAVE_DIR}${DATASETS[$d]}
    done
done
