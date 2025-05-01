from lib.dataset import load_countdown_dataset
from transformers import AutoTokenizer

def test_load_dataset():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    train_dataset, test_dataset = load_countdown_dataset(tokenizer, size=100)
    assert isinstance(train_dataset[0]["target"], int)
    assert isinstance(train_dataset[0]["nums"], list)
    assert all(isinstance(n, int) for n in train_dataset[0]["nums"])
