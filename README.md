# Platypus: Quick, Cheap, and Powerful Refinement of LLMs (https://platypus-llm.github.io)

<p align="center">
<img src="./assets/Best_Platty.png" alt="Platypus" width="300"/>
</p>

The Platypus models are a series of fine-tuned and merged variants based on the LLaMA and LLaMa-2 transformer architectures. Platypus takes advantage of [LoRA](https://arxiv.org/pdf/2106.09685.pdf) and [PEFT](https://github.com/huggingface/peft). 

All models and dataset available via HuggingFace: [`garage-bAInd`](https://huggingface.co/garage-bAInd)

## Updates
**8/14/23**: We have cleaned up our pipeline and added data refinement and similarity code. Within in the next few days we'll have a script to reproduce our exact dataset from 11 open-source datasets.

**8/13/23**: An unquantized GPU chatbot of OpenOrca-Platypus2-13B, our most recent collab, is available via Hugging Face spaces, courtesy of OpenOrca: [Chat now!](https://huggingface.co/spaces/Open-Orca/OpenOrca-Platypus2-13B)

<p align="center">
<img src="./assets/orca_platty.jpeg" alt="Platypus" width="120"/>
</p>

**8/11/23**: Our [paper](https://arxiv.org/abs/2308.07317) and [project website](https://platypus-llm.github.io) have been released!

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

Note: The script above uses `torchrun` for data parallelism. PyTorch is not in `requirements.txt` since technically you can run fine-tuning without it (after a few minor changes to the .py file). To use `fine-tuning.sh`, please install [PyTorch](https://pytorch.org/get-started/locally/). We recommend using `torchrun` and PyTorch 2.0+ for speed + `torch.compile`. If you do not install pytorch, or use an alternative method like `accelerate launch`, please take time to comment out any torch related lines in the scirpts.

Hyperparameters used to fine-tune Platypus:

| Hyperparameter      | Value 13B / 70B  |
|---------------------|--------|
| learning rate       | 4e-4 / 3e-4   |
| batch size          | 16     |
| microbatch  size    | 1      |
| warmup steps        | 100    |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | cosine |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| lora target modules | gate_proj, up_proj, down_proj|
| cutoff length       | 4096   |
| train on inputs     | False  |
| group by length     | False  |
| add eos token       | False  |

Example for how to calcualte gradient accumulation steps using 2 GPUs: = global_batch_size / micro_batch_size / num_gpus = 16 / 1 / 2 = 8.

If your model **cannot** fit on the memory of each GPU, please use the alternative fine-tuning option below (or utilize accelerate, FDSP, etc.) to take advantage of model parallelism. A good alternative to torchrun is accelerate. 

```bash
python finetune.py \
    --base_model meta-llama/Llama-2-70b-hf \
    --data-path ./final_data.json \
    --output_dir ./llama2-platypus-70b \
    --batch_size 16 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 0.0003 \
    --cutoff_len 4096 \
    --val_set_size 0 \
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
```

## Merging

Once you've completed a fine-tuning, use `merge.sh` to merge the LoRA weights back into the base LLaMa model (or base model of your choice) for export to HuggingFace format.

While we are experimenting on better and alternative ways to merge (stay tuned!), our current merging process relies on the basic linear merge provided by PEFT. Before we fine-tune, we search for possible models to merge with and the datasets used to create them (to the best of our ability). The success of our LoRA merges stems from using the right data. Our most successful merges have little to no overlap in fine-tuning data. For example, GPlatty-30B is a merge of Platypus-30B and gpt4-alpaca-lora-30b. We saw a 2% jump in accuracy for GPlatty, and the datasets used to fine-tune the aforementioned two LoRA-based models had very low similarity scores. Please see [our paper](https://arxiv.org/abs/2308.07317) for additional information. 

**NOTE:** If you encounter any errors while merging, please try uninstalling bitsandbytes and peft, then reinstalling with the newest versions (peft should always be installed from source).

## Dataset Refinement

We used keyword search to find STEM and logic questions in the 11 open-source datasets that make up [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus). Then, to remove duplicates and redundancy, we perform a cosine similarity check of the questions using SentenceTransformers embeddings. Lastly, we do a similarity check to remove any questions from our training set that are too similiar to the test set.

You can access all of the related code in the `data_pipeline` folder of this repo.

## Reproducing Benchmark Eval Results
Install LM Evaluation Harness:
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463 # The commit used by the Open LLM Leaderboard
pip install -e .
```
Each task was evaluated on a single A100 80GB GPU for 13B, and 2 A100s for 70B.

ARC:
```
python main.py --model hf-causal-experimental --model_args pretrained=garage-bAInd/Platypus-13B,use_accelerate=True --tasks arc_challenge --batch_size 2 --no_cache --write_out --output_path results/Platypus-13B/arc_challenge_25shot.json --device cuda --num_fewshot 25
```

HellaSwag:
```
python main.py --model hf-causal-experimental --model_args pretrained=garage-bAInd/Platypus-13B,use_accelerate=True --tasks hellaswag --batch_size 2 --no_cache --write_out --output_path results/Platypus-13B/hellaswag_10shot.json --device cuda --num_fewshot 10
```

MMLU:
```
python main.py --model hf-causal-experimental --model_args pretrained=garage-bAInd/Platypus-13B,use_accelerate=True --tasks hendrycksTest-* --batch_size 2 --no_cache --write_out --output_path results/Platypus-13B/mmlu_5shot.json --device cuda --num_fewshot 5
```

TruthfulQA:
```
python main.py --model hf-causal-experimental --model_args pretrained=garage-bAInd/Platypus-13B,use_accelerate=True --tasks truthfulqa_mc --batch_size 2 --no_cache --write_out --output_path results/Platypus-13B/truthfulqa_0shot.json --device cuda
```
## Inference for Adapters (`inference.py`)

This a basic example script for running inference directly using fine-tuned adapters and/or local data. The current version reads data from a csv file. You can easily edit this to pull from HF or use a json file. Please make any necessary edits before using this script (it assumes alpaca formatting).

## BibTeX

```
@article{platypus2023,
    title={Platypus: Quick, Cheap, and Powerful Refinement of LLMs}, 
    author={Ariel N. Lee and Cole J. Hunter and Nataniel Ruiz},
    booktitle={arXiv preprint arxiv:2308.07317},
    year={2023}
}
```