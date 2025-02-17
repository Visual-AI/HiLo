#!/bin/bash
PYTHON="/disk/work/hjwang/miniconda3/envs/hilo/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

model='dino'

DATASETS=("domainnet")

ENVs1=("real")
ENVs2=("real" "clipart")

TASKs=("A_L+A_U+B->A_U+B+C")

partial_dataset='False'
uda_loss='True'

SAVE_DIR=/disk/work/hjwang/HiLo/logs/mi_pmtrans/
WEIGHTS_PATH=/disk/work/hjwang/HiLo/data/domainnet/clipart.json

for d in ${!DATASETS[@]}; do
    for t in ${!TASKs[@]}; do
        for e in ${!ENVs1[@]}; do
            for ee in ${!ENVs2[@]}; do
                if [ ${ENVs1[$e]} != ${ENVs2[$ee]} ]; then
                    echo ${ENVs1[$e]} ${ENVs2[$ee]} others
                    ${PYTHON} -m methods.PMTrans.mi_dis_pm \
                                --dataset_name 'domainnet' \
                                --batch_size 128 \
                                --grad_from_block 11 \
                                --epochs 200 \
                                --num_workers 8 \
                                --sup_weight 0.35 \
                                --weight_decay 5e-5 \
                                --warmup_teacher_temp 0.07 \
                                --teacher_temp 0.04 \
                                --warmup_teacher_temp_epochs 30 \
                                --memax_weight 2 \
                                --transform 'domainnet' \
                                --weights_path ${WEIGHTS_PATH} \
                                --lr 0.05 \
                                --eval_funcs 'v2' \
                                --src_env ${ENVs1[$e]} \
                                --aux_env ${ENVs2[$ee]} \
                                --task_type ${TASKs[$t]} \
                                --use_partial_dataset ${partial_dataset} \
                                --use_uda_loss ${uda_loss} \
                                --model ${model} \
                                --model_path ${SAVE_DIR}${ENVs1[$e]}-${ENVs2[$ee]}-others
                else
                    continue
                fi
            done
        done
    done
done
