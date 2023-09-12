## Fine-tune Llama-7b

1. `conda create -n platypus-3.9 python=3.9`
2. `pip install -r requirements.txt`
3. `huggingface-cli login` to be able to download llama from hf
4. `wandb login` if needed
5. `conda install cudatoolkit` to solve `UserWarning: WARNING: No libcudart.so found!` ([link](https://stackoverflow.com/questions/70967651/could-not-load-dynamic-library-libcudart-so-11-0))
6. apply [this](https://github.com/arielnlee/Platypus#updates) to `finetune.py`
7. change `base_model` in `fine-tune.sh` to `meta-llama/Llama-2-7b-hf`
8. you're ready to go! 🚀 run `./fine-tune.sh`

extras:

9. `export LOCAL_RANK=0` to let the script know which is the main process among all gpus
10. `export WORLD_SIZE=2` to let the script know there are 2 gpus (not needed actually, as we pass this param to `torchrun`)
11. increase `val_set_size` in `fine-tune.sh` to usa a validation set
12. add `--wandb_project platypus` in `fine-tune.sh` to log to the correct wandb project
13. change the code in `finetune.py` to log a meaningful `run_name` to wandb
```python
# inside the if LOCAL_RANK == 0 block
use_wandb = len(wandb_project) > 0
if use_wandb:
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{base_model}_batch-{batch_size}_{now_str}"
    wandb.init(project=wandb_project, name=run_name)
```
 
## Merge the LoRA adapter into the original model
1. use `merge.sh` modifying the params as needed
```shell
python merge.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --peft_model_path ./llama2-platypus-7b/checkpoint-1000 \
    --output_dir ./llama2-7b-merged
```

## Evaluate the merged model
1. install FastChat `pip3 install "fschat[model_worker,webui]"`
2. run the merged model `python -m fastchat.serve.cli --model-path ./llama2-7b-merged --num-gpus 2`

## Push the merged model to HF
1. `huggingface-cli login` if needed
2. `python push_to_hub.py --model_name llama2-7b-platypus-ckpt1000`

## Fine-tune, merge and push the 13b model
1. the same approach is used, just change the paths and the model names
2. also don't apply the `finetune.py` patch (step 6) while fine-tuning
