from transformers import AutoModel, AutoTokenizer
from fire import Fire
from huggingface_hub import create_repo


def push(model_name):
    repo_name = f"bsp-albz/{model_name}"
    create_repo(repo_name, private=False)

    # load model and tokenizer from local file
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # push them to the hub
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)


if __name__ == "__main__":
    Fire(push)
