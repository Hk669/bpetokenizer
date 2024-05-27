import os
import pytest
from bpetokenizer import BPETokenizer

@pytest.fixture
def tokenizer():
    return BPETokenizer()


def test_train():
    """Test the training of the tokenizer."""
    text = "aaabdaaabac"
    tokenizer = BPETokenizer()
    tokenizer.train(text, 259, verbose=False)
    assert len(tokenizer.vocab) == 259
    assert len(tokenizer.merges) == 3
    assert tokenizer.decode(tokenizer.encode(text)) == "aaabdaaabac"


def test_encode():
    """Test the encoding of the tokenizer."""
    text = "aaabdaaabac"
    tokenizer = BPETokenizer()
    tokenizer.train(text, 259, verbose=False)
    assert tokenizer.encode("aaabdaaabac") == [258, 100, 258, 97, 99]


def test_decode():
    """Test the decoding of the tokenizer."""
    text = "aaabdaaabac"
    tokenizer = BPETokenizer()
    tokenizer.train(text, 259, verbose=False)
    assert tokenizer.decode([258, 100, 258, 97, 99]) == "aaabdaaabac"