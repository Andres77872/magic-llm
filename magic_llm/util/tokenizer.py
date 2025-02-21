from tokenizers import Tokenizer, Encoding


def from_file(model, text) -> Encoding:
    tokenizer = Tokenizer.from_file(model)
    encoded = tokenizer.encode(text)
    return encoded


def from_hf(model, text) -> Encoding:
    tokenizer = Tokenizer.from_pretrained(model)
    encoded = tokenizer.encode(text)
    return encoded
