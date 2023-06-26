# 🥳 Platypus-30b

Platypus-30B is an instruction fine-tuned model based on the LLaMA-30b transformer architecture. Availble via HuggingFace: [`lilloukas/Platypus-30b`](https://huggingface.co/lilloukas/Platypus-30b).

| Benchmark Metric      | Value |
|-----------------------|-------|
| MMLU (5-shot)         | 65.4  |
| ARC (25-shot)         | 64.6  |
| HellaSwag (10-shot)   | 84.3  |
| TruthfulQA (0-shot)   | 45.8  |
| Avg.                  | 65 💥 |

Platypus-30B also scored 70.8 (10th out of 49 models) on the [ReClor](https://eval.ai/web/challenges/challenge-page/503/leaderboard/1347) test set.

### Local Setup

This repository is multi-GPU friendly, and provides code to use model OR data parellelism, depending on your computational resources. 

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Be sure to use these exact requirements or you may run into model saving or OOM issues.

### Fine-tuning (`finetune.py`)

Example 1: For smaller models when using ONE GPU, as long as the size of the model is less than the GPU RAM (i.e. run 13B on 1 A100 40GB).

```bash
python finetune.py \
    --base_model '../llama13B_hf' \
    --data_path '../data-test.csv' \
    --output_dir '' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 5 \
    --learning_rate 4e-4 \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs False \
    --group_by_length False
```

Example 2: For use with multiple GPUS, if you want to use torchrun AND model size is less than RAM of each individual GPU (i.e. run 30B on 4 A100 80GB). This will create a copy of the model on each GPU, so may cause OOM for large models depending on your hardware.

```bash
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 finetune.py \
    --base_model '../llama30B_hf' \
    --data_path '../data-test.csv' \
    --output_dir '' \
```

Example 3: For use with multiple GPUS, if you don't want to use torchrun OR model size is greater than RAM of each individual GPU (i.e. run 65B on 8 A100 80GB, 8 A6000 48GB, etc.). This will allow model parallelism and is best for LARGE models like 65B. 

```bash
export WORLD_SIZE=1

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py \
    --base_model '../llama65B_hf' \
    --data_path '../data-test.csv' \
    --output_dir '' \
```

### Inference (`inference.py`)

Run inference using a csv or json file. Inference commands follow the same structure noted above for fine-tuning.

### Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).
