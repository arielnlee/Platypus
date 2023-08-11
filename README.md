# 🥳 Platypus

The Platypus models are a series of fine-tuned variants based on the LLaMA and LLaMa 2 transformer architectures. Platty takes advantage of [LoRA](https://arxiv.org/pdf/2106.09685.pdf). 

All models available via HuggingFace: [`garage-bAInd`](https://huggingface.co/garage-bAInd)

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
python3 -m fastchat.serve.cli --model-path garage-bAInd/Platypus-30B --conv_template alpaca
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

Note: The script above uses `torchrun` for data parallelism. PyTorch is not in `requirements.txt` since technically you can run fine-tuning without it. To use `fine-tuning.sh`, please install [PyTorch](https://pytorch.org/get-started/locally/). We recommend using `torchrun` and PyTorch 2.0+ for speed + `torch.compile`. If you do not install pytorch, please take time to comment out any torch related lines in the scirpts.

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
## Reproducing Evaluation Results
Install LM Evaluation Harness:
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463 # The commit used by the Open LLM Leaderboard
pip install -e .
```
Each task was evaluated on a single A100 80GB GPU.

ARC:
```
python main.py --model hf-causal-experimental --model_args pretrained=garage-bAInd/Platypus-30B --tasks arc_challenge --batch_size 1 --no_cache --write_out --output_path results/Platypus-30B/arc_challenge_25shot.json --device cuda --num_fewshot 25
```

HellaSwag:
```
python main.py --model hf-causal-experimental --model_args pretrained=garage-bAInd/Platypus-30B --tasks hellaswag --batch_size 1 --no_cache --write_out --output_path results/Platypus-30B/hellaswag_10shot.json --device cuda --num_fewshot 10
```

MMLU:
```
python main.py --model hf-causal-experimental --model_args pretrained=garage-bAInd/Platypus-30B --tasks hendrycksTest-* --batch_size 1 --no_cache --write_out --output_path results/Platypus-30B/mmlu_5shot.json --device cuda --num_fewshot 5
```

TruthfulQA:
```
python main.py --model hf-causal-experimental --model_args pretrained=garage-bAInd/Platypus-30B --tasks truthfulqa_mc --batch_size 1 --no_cache --write_out --output_path results/Platypus-30B/truthfulqa_0shot.json --device cuda
```
## Inference (`inference.py`)

Run inference using a csv or json file. Inference commands follow the same structure noted above for fine-tuning.

### Checkpoint export (`export_*_checkpoint.py`)

These files contain scripts that merge the LoRA weights back into the base model
for export to HuggingFace format and to PyTorch `state_dicts`.
They should help users
who want to run inference in projects like [llama.cpp](https://github.com/ggerganov/llama.cpp)
or [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).
