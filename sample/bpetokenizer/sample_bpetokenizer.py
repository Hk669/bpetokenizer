import sys
sys.path.append("../")

from bpetokenizer import BPETokenizer

special_tokens = {
    "<|endoftext|>": 1001,
    "<|startoftext|>": 1002,
    "[SPECIAL1]": 1003,
    "[SPECIAL2]": 1004,
}

tokenizer = BPETokenizer(special_tokens=special_tokens)
texts = "<|startoftext|> Hello, World! This is a sample text with the special tokens [SPECIAL1] and [SPECIAL2] to test the tokenizer.<|endoftext|>"

tokenizer.train(texts, vocab_size=310, verbose=True)

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