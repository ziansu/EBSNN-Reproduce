#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

labels=reddit,facebook,NeteaseMusic,twitter,qqmail,instagram,weibo,iqiyi,
labels+=imdb,TED,douban,amazon,youtube,JD,youku,baidu,google,tieba,taobao,bing


model_dir=../save/teacher
restore_file=checkpoint-best-epoch_42-acc_0.9986.pt
output_dir=../save/student

python train_kd.py \
    --do_train \
    --data_dir ../data \
    --dataset d1 \
    --output_dir ${output_dir} \
    --epochs 30 --labels $labels \
    --segment_len 16 \
    --max_length 1500 \
    --batch_size 256 \
    --learning_rate 0.01 \
    --kd_alpha 0.9 \
    --kd_temperature 0.5 \
    --focal \
    --student_config ../save/student/config_small.json \
    --teacher_config ../save/teacher/teacher_config.json \
    --model_dir $model_dir \
    --restore_file $restore_file \
    --log_filename ${output_dir}/plog.log \
    --logging_steps 50 \
    --shuffle