# 🥳 Platypus-30B

Platypus-30B is an instruction fine-tuned model based on the LLaMA-30B transformer architecture. Availble via HuggingFace: [`lilloukas/Platypus-30b`](https://huggingface.co/lilloukas/Platypus-30b).

| Benchmark Metric      | Value |
|-----------------------|-------|
| MMLU (5-shot)         | 65.4  |
| ARC (25-shot)         | 64.6  |
| HellaSwag (10-shot)   | 84.3  |
| TruthfulQA (0-shot)   | 45.8  |
| Avg.                  | 65 💥 |

Platypus-30B also acheived an accuracy of 70.8 (10th out of 49 models) on the [ReClor](https://eval.ai/web/challenges/challenge-page/503/leaderboard/1347) test set.

We have also successfully run a fine-tuning of LlaMa-65B using this repository. 

### Local Setup

This repository is multi-GPU friendly, and provides code to use model OR data parellelism, depending on your computational resources. 

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Be sure to use these exact requirements or you may run into model saving or OOM issues.

### Fine-tuning (`finetune.py`)

Run `fine-tuning.sh`.

Note: The script above uses torchrun. PyTorch is not in `requirements.txt` since technically you can run the repository without it. To use the script above, please install [PyTorch](https://pytorch.org/get-started/locally/). We recommend using torchrun, since it's faster, and PyTorch 2.0+ for `torch.compile`.

Hyperparameters used to fine-tune Platypus-30B follow:

| Hyperparameter      | Value |
|---------------------|-------|
| learning_rate       | ---   |
| batch_size          | ---   |
| microbatch_size     | ---   |
| warmup_steps        | ---   |
| epochs              | ---   |
| weight_decay        | ---   |
| optimizer           | ---   |
| weight_decay        | ---   |
| cutoff_len          | ---   |
| lora_target_modules | ---   |

If your model cannot fit on the memory of each GPU, please see the alternative training option below for model parallelism.

```bash
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0,1

python finetune.py \
    --base_model './llama13B_hf' \
    --data_path './train_final.json' \
    --output_dir './Platypus-30b' \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 2e-4 \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs False \
    --group_by_length False
```
### Inference (`inference.py`)

Run inference using a csv or json file. Inference commands follow the same structure noted above for fine-tuning.

### Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to Hugging Face format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).
