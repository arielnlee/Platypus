python push_to_hub.py \
    --base_model_name_or_path hyunseoki/ko-en-llama2-13b \
    --peft_model_path persona/checkpoint-60 \
    --output_dir persona-llama2-13b \
    --push True