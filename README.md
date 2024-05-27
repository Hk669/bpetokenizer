# bpetokenizer

A Byte Pair Encoding (BPE) tokenizer, which algorithmically follows along the GPT tokenizer. The tokenizer is capable of handling special tokens and uses a customizable regex pattern for tokenization(includes the gpt4 regex pattern).


### Overview

The Byte Pair Encoding (BPE) algorithm is a simple yet powerful method for building a vocabulary of subword units for a given text corpus. This tokenizer can be used for training your tokenizer of the LLM on various languages of text corpus.

this algorithm is first introduced in the paper [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909) and then used this in the gpt2 tokenizer([Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))

The [notebook](notebooks/tokenization.ipynb) which shows the BPE algorithm in detail and how the tokenizers work internally.

Every LLM(LLama, Gemini, Mistral..) use their own Tokenizers trained on their own text dataset.


### Features

- Implements Byte Pair Encoding (BPE) algorithm.
- Handles special tokens.
- Uses a customizable regex pattern for tokenization.
- Compatible with Python 3.9 and above


#### This repository has 2 different Tokenizers:
- `BPETokenizer`
- `Tokenizer`

1. [Tokenizer](bpetokenizer/base.py): This class contains `train`, `encode`, `decode` and functionalities to `save` and `load`. Also contains few helper functions `get_stats`, `merge`, `replace_control_characters`..  to perform the BPE algorithm for the tokenizer.

2. [BPETokenizer](bpetokenizer/tokenizer.py): This class emphasizes the real power of the tokenizer(used in gpt4 tokenizer..[tiktoken](https://github.com/openai/tiktoken)), uses the `GPT4_SPLIT_PATTERN` to split the text as mentioned in the gpt4 tokenizer. also handles the `special_tokens` (refer [sample_bpetokenizer](sample/bpetokenizer/sample_bpetokenizer.py)). which inherits the `save` and `load` functionlities to save and load the tokenizer respectively.


### Usage

this tutorial leverages the `special_tokens` usage in the Tokenizer.

Install the package

```shell
pip install bpetokenizer
```


```py
from bpetokenizer import BPETokenizer

special_tokens = {
    "<|endoftext|>": 1001,
    "<|startoftext|>": 1002,
    "[SPECIAL1]": 1003,
    "[SPECIAL2]": 1004,
}

tokenizer = BPETokenizer(special_tokens=special_tokens) # you can also use the method _special_tokens to register the special tokens (if not passed when intializing)
texts = "<|startoftext|> Hello, World! This is a sample text with the special tokens [SPECIAL1] and [SPECIAL2] to test the tokenizer.<|endoftext|>"

tokenizer.train(texts, vocab_size=310, verbose=True)
# tokenizer._special_tokens(special_tokens) # if not passed when intialization of the BPETokenizer

encode_text = """
<|startoftext|>Hello, World! This is a sample text with the special tokens [SPECIAL1] and [SPECIAL2] to test the tokenizer.
Hello, Universe! Another example sentence containing [SPECIAL1] and [SPECIAL2], used to ensure tokenizer's robustness.
Greetings, Earth! Here we have [SPECIAL1] appearing once again, followed by [SPECIAL2] in the same sentence.
Hello, World! This is yet another sample text, with [SPECIAL1] and [SPECIAL2] making an appearance.
Hey there, World! Testing the tokenizer with [SPECIAL1] and [SPECIAL2] to see if it handles special tokens properly.
Salutations, Planet! The tokenizer should recognize [SPECIAL1] and [SPECIAL2] in this long string of text.
Hello again, World! [SPECIAL1] and [SPECIAL2] are special tokens that need to be handled correctly by the tokenizer.
Welcome, World! Including [SPECIAL1] and [SPECIAL2] multiple times in this large text to ensure proper encoding.
Hi, World! Let's add [SPECIAL1] and [SPECIAL2] in various parts of this long sentence to test the tokenizer thoroughly.
<|endoftext|>
"""
ids = tokenizer.encode(encode_text, special_tokens="all")
print(ids)

decode_text = tokenizer.decode(ids)
print(decode_text)

tokenizer.save("sample_bpetokenizer")
```

refer [sample_bpetokenizer](sample/bpetokenizer) to have an understanding of the `vocab` and the `model` file of the tokenizer trained on the above texts.


### Run Tests

the tests folder `tests/` include the tests of the tokenizer, uses pytest.

```
python3 -m pytest
```

additionally, the workflows are setup to run the tests when made a PR.


### Contributing

Contributions to the BPE Tokenizer are most welcomed! If you would like to contribute, please follow these steps:

- Star and Fork the repository.
- Create a new branch (git checkout -b feature/your-feature).
- Commit your changes (git commit -am 'Add some feature').
- Push to the branch (git push origin feature/your-feature).
- Create a new Pull Request.

Please ensure your code follows the project's coding standards and includes appropriate tests. Also, update the documentation as necessary.


### License

This project is licensed under the MIT License.

----

*this tokenizer is inspired from the [minbpe](https://github.com/karpathy/minbpe), but more optimized.