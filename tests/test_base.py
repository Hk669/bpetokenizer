import pytest
from bpetokenizer import Tokenizer


@pytest.fixture()
def tokenizer():
    text = "<|start|>This is a test text for training the vocab of the tokenizer<|end|>"
    special_tokens = {
        "<|start|>" : 1001,
        "<|end|>": 1002
    }
    tokenizer = Tokenizer(special_tokens=special_tokens)
    tokenizer.train(text, vocab_size=270, min_frequency=0)
    return tokenizer

def test_train():
    text = "<|start|>This is a test text for training the vocab of the tokenizer<|end|>"
    special_tokens = {
        "<|start|>" : 1001,
        "<|end|>": 1002
    }
    tokenizer = Tokenizer(special_tokens=special_tokens)
    tokenizer.train(text, vocab_size=270, min_frequency=0)
    assert tokenizer.encode(text)
    assert len(tokenizer.vocab) == 270
    assert len(tokenizer.merges) == 270 - 256
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_encode(tokenizer):
    """Test encoding with different text lengths and special tokens."""

    # Test with short text
    short_text = "hello"
    encoded_short = tokenizer.encode(short_text)
    assert len(encoded_short) > 0  # Encoded text should not be empty

    # Test with long text
    long_text = "This is a very long text to test the tokenizer's encoding capabilities."
    encoded_long = tokenizer.encode(long_text)
    assert len(encoded_long) > 0  # Encoded text should not be empty

    # Test with special tokens
    special_text = "<|start|>This has special tokens<|end|>"
    tokenizer.train(special_text, vocab_size=260, min_frequency=0)
    encoded_special = tokenizer.encode(special_text)
    assert all(t in tokenizer.vocab for t in encoded_special)  # All tokens should be in vocab


def test_decode(tokenizer):
    """Test decoding functionality with different encoded inputs."""

    encoded_text = [1, 2, 3]
    decoded_text = tokenizer.decode(encoded_text)
    assert len(decoded_text) > 0
