python inference.py \
    --base_model hyunseoki/ko-en-llama2-13b \
    --lora_weights persona/checkpoint-60 \
    --csv_path test_persona.csv \
    --output_csv_path output.csv