import os
import pytest
from bpetokenizer import BPETokenizer, Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer()

@pytest.fixture
def bpe_tokenizer():
    return BPETokenizer()


def test_train():
    """Test the training of the tokenizer."""
    text = "aaabdaaabac"
    tokenizer = Tokenizer()
    tokenizer.train(text, 259, verbose=False)
    assert len(tokenizer.vocab) == 259
    assert len(tokenizer.merges) == 3
    assert tokenizer.decode(tokenizer.encode(text)) == "aaabdaaabac"


def test_encode():
    """Test the encoding of the tokenizer."""
    text = "aaabdaaabac"
    tokenizer = Tokenizer()
    tokenizer.train(text, 259, verbose=False)
    assert tokenizer.encode("aaabdaaabac") == [258, 100, 258, 97, 99]


def test_decode():
    """Test the decoding of the tokenizer."""
    text = "aaabdaaabac"
    tokenizer = Tokenizer()
    tokenizer.train(text, 259, verbose=False)
    assert tokenizer.decode([258, 100, 258, 97, 99]) == "aaabdaaabac"


def test_train_bpe():
    """Test the training of the BPE tokenizer."""
    text = "aaabdaaabac"
    tokenizer = BPETokenizer()
    tokenizer.train(text, 256 + 3, verbose=False)
    assert len(tokenizer.vocab) == 259
    assert len(tokenizer.merges) == 3
    assert tokenizer.decode(tokenizer.encode(text)) == "aaabdaaabac"


def test_train_bpe_w_special_tokens():
    """Test the bpetokenizer with special tokens"""
    special_tokens = {
        "<|endoftext|>": 1001,
        "<|startoftext|>": 1002,
        "[SPECIAL1]": 1003,
        "[SPECIAL2]": 1004,
    }

    PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    tokenizer = BPETokenizer(special_tokens=special_tokens, pattern=PATTERN)
    texts = "<|startoftext|> Hello, World! This is a sample text with the special tokens [SPECIAL1] and [SPECIAL2] to test the tokenizer.<|endoftext|>"
    tokenizer.train(texts, vocab_size=310, verbose=False)

    assert len(tokenizer.vocab) == 310
    assert len(tokenizer.merges) == 310 - 256
    assert tokenizer.decode(tokenizer.encode(texts)) == texts
    assert tokenizer.inverse_special_tokens == {v: k for k,v in special_tokens.items()}
    assert tokenizer.special_tokens == special_tokens
    assert tokenizer.pattern == PATTERN
