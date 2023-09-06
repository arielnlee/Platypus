## Finetuning Llama-7b

- `conda create -n platypus-3.9 python=3.9`
- `pip install -r requirements.txt`
- `huggingface-cli login` to be able to download llama from hf
- `wandb login` if needed
- `conda install cudatoolkit` to solve `UserWarning: WARNING: No libcudart.so found!` ([link](https://stackoverflow.com/questions/70967651/could-not-load-dynamic-library-libcudart-so-11-0))
- apply [this](https://github.com/arielnlee/Platypus#updates) to `finetune.py`
- change `base_model` in `fine-tune.sh` to `meta-llama/Llama-2-7b-hf`
- you're ready to go! 🚀 run `./fine-tune.sh`

extras:
- `export LOCAL_RANK=0` to let the script know which is the main process among all gpus
- increase `val_set_size` in `fine-tune.sh` to usa a validation set
- add `--wandb_project platypus` in `fine-tune.sh` to log to the correct wandb project
- change the code in `finetune.py` to log a meaningful `run_name` to wandb
```python
# inside the if LOCAL_RANK == 0 block
use_wandb = len(wandb_project) > 0
if use_wandb:
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{base_model}_batch-{batch_size}"
    wandb.init(project=wandb_project, name=run_name, group=now_str)
```
- 
