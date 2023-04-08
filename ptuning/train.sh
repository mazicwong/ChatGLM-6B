PRE_SEQ_LEN=128
LR=2e-2

#export CUDA_VISIBLE_DEVICES=1,6
python main.py \
    --do_train \
    --train_file /apdcephfs/share_1443437/zhiqihuang/backbone/data/AdvertiseGen/train.json \
    --validation_file /apdcephfs/share_1443437/zhiqihuang/backbone/data/AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /apdcephfs/share_1443437/zhiqihuang/glm \
    --output_dir /apdcephfs/share_1443437/zhiqihuang/ChatGLM-6B/output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN
#     \
#    --quantization_bit 4

