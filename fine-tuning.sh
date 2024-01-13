python finetune.py \
    --base_model hyunseoki/ko-en-llama2-13b \
    --data-path output_files.jsonl \
    --output_dir ./persona \
    --batch_size 16 \
    --micro_batch_size 8 \
    --num_epochs 10 \
    --learning_rate 0.0004 \
    --cutoff_len 2048 \
    --val_set_size 10 \
    --eval_steps 10 \
    --save_steps 10 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --warmup_steps 100
