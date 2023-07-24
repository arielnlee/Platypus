# 🥳 Platypus-30B

Platypus-30B is an instruction fine-tuned model based on the LLaMA-30B transformer architecture. Platty takes advantage of [LoRA](https://arxiv.org/pdf/2106.09685.pdf). 

Availble via HuggingFace: [`garage-bAInd/Platypus-30B`](https://huggingface.co/garage-bAInd/Platypus-30B)

| Metric                | Value |
|-----------------------|-------|
| MMLU (5-shot)         | 64.2  |
| ARC (25-shot)         | 64.6  |
| HellaSwag (10-shot)   | 84.3  |
| TruthfulQA (0-shot)   | 45.8  |
| Avg.                  | 64.7 💥 | 


We use state-of-the-art EleutherAI [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to run the benchmark tests above.

Platypus-30B acheives an accuracy of 70.8 on the [ReClor](https://whyu.me/reclor/) test set.

We have also successfully run a fine-tuning of LlaMa-65B using this repository. 

## Platty Merges

### SuperPlatty-30B

SuperPlatty-30B is a merge of [`garage-bAInd/Platypus-30B`](https://huggingface.co/garage-bAInd/Platypus-30B) and [`kaiokendev/SuperCOT-LoRA`](https://huggingface.co/kaiokendev/SuperCOT-LoRA). 

Available via HuggingFace: [`garage-bAInd/SuperPlatty-30B`](https://huggingface.co/garage-bAInd/SuperPlatty-30B)

| Metric                | Value |
|-----------------------|-------|
| MMLU (5-shot)         | 62.6  |
| ARC (25-shot)         | 66.1  |
| HellaSwag (10-shot)   | 83.9  |
| TruthfulQA (0-shot)   | 54.0  |
| Avg.                  | 66.6  | 

### GPlatty-30B

GPlatty-30B is a merge of [`garage-bAInd/Platypus-30B`](https://huggingface.co/garage-bAInd/Platypus-30B) and [`chansung/gpt4-alpaca-lora-30b`](https://huggingface.co/chansung/gpt4-alpaca-lora-30b). 

Available via HuggingFace: [`garage-bAInd/GPlatty-30B`](https://huggingface.co/garage-bAInd/GPlatty-30B)

| Metric                | Value |
|-----------------------|-------|
| MMLU (5-shot)         | 63.6  |
| ARC (25-shot)         | 66    |
| HellaSwag (10-shot)   | 84.8  |
| TruthfulQA (0-shot)   | 53.8  |
| Avg.                  | 67 🔥 | 


## CLI 

[Fastchat](https://github.com/lm-sys/FastChat) provides a simple setup for those interested in running the model. Afrer downloading the model through HuggingFace, clone the Fastchat repository:

```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
```

Download the required packages:

```
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e .
```

Finally, run the following:

```
python3 -m fastchat.serve.cli --model-path lilloukas/Platypus-30B --conv_template alpaca
```

## Local Setup

This repository is multi-GPU friendly, and provides code to use model or data parellelism, depending on your computational resources. 

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Be sure to use these exact requirements or you may run into model saving or OOM issues.

## Fine-tuning (`finetune.py`)

Run `fine-tuning.sh`.

Note: The script above uses `torchrun` for data parallelism. PyTorch is not in `requirements.txt` since technically you can run fine-tuning without it. To use `fine-tuning.sh`, please install [PyTorch](https://pytorch.org/get-started/locally/). We recommend using `torchrun` and PyTorch 2.0+ for speed + `torch.compile`.

Hyperparameters used to fine-tune Platypus-30B:

| Hyperparameter      | Value  |
|---------------------|--------|
| learning rate       | 4e-4   |
| batch size          | 128    |
| microbatch  size    | 8      |
| warmup ratio        | 0.03   |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | cosine |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| lora target modules | q_proj,k_proj,v_proj,o_proj|
| cutoff length       | 2048   |
| train on inputs     | False  |
| group by length     | False  |
| add eos token       | False  |

Gradient accumulation steps = global_batch_size / micro_batch_size / num_gpus = 128 / 8 / 4 = 4.

If your model **cannot** fit on the memory of each GPU, please use the alternative fine-tuning option below to take advantage of model parallelism.

```bash
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

python finetune.py \
    --base_model './llama30B_hf' \
    --data_path './train_final.json' \
    --output_dir './Platypus-30b' \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 1 \
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
## Inference (`inference.py`)

Run inference using a csv or json file. Inference commands follow the same structure noted above for fine-tuning.

### Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to HuggingFace format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).
