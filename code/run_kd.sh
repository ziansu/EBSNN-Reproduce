#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

labels=reddit,facebook,NeteaseMusic,twitter,qqmail,instagram,weibo,iqiyi,
labels+=imdb,TED,douban,amazon,youtube,JD,youku,baidu,google,tieba,taobao,bing


model_dir=.
restore_file=checkpoint-best
output_dir=../save/student

python run.py \
    --do_train \
    --data_dir ../data \
    --dataset d1 \
    --output_dir ${output_dir} \
    --epochs 5 --labels $labels \
    --batch_size 256 --gpu 0 --gamma 1 \
    --student_config student.json \
    --teacher_config teacher.json \
    --model_dir $model_dir \
    --restore_file $restore_file \
    --log_filename ${output_dir}/plog.log \
    --logging_steps 50 \
    --shuffle