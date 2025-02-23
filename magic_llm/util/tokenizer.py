import tiktoken
from tokenizers import Tokenizer, Encoding


def from_file(model, text) -> Encoding:
    tokenizer = Tokenizer.from_file(model)
    encoded = tokenizer.encode(text)
    return encoded


def from_hf(model, text) -> Encoding:
    tokenizer = Tokenizer.from_pretrained(model)
    encoded = tokenizer.encode(text)
    return encoded


def from_openai(text, model='gpt-4o') -> list[int]:
    encoding = tiktoken.encoding_for_model(model)
    encoded = encoding.encode(text)
    return encoded
