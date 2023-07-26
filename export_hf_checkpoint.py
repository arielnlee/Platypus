import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer 

# Specify the device to use
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_MODEL = "meta-llama/Llama-2-70b-hf"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    #device_map={"": device},
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)},
)
#).to(device)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    "./llama2-platypus-70b",
    #device_map={"": device},
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)},
    torch_dtype=torch.float16,
)
#).to(device)

lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, "./latypus2-70B", state_dict=deloreanized_sd, max_shard_size="5000MB"
)
