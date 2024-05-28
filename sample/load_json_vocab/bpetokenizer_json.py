from bpetokenizer import BPETokenizer

tokenizer = BPETokenizer()

tokenizer.load("sample_bpetokenizer.json", mode="json")

encode_text = """
<|startoftext|>Hello, World! This is a sample text with the special tokens [SPECIAL1] and [SPECIAL2] to test the tokenizer.
Hello, Universe! Another example sentence containing [SPECIAL1] and [SPECIAL2], used to ensure tokenizer's robustness.
Greetings, Earth! Here we have [SPECIAL1] appearing once again, followed by [SPECIAL2] in the same sentence.<|endoftext|>"""

print("vocab: ", tokenizer.vocab)
print('---')
print("merges: ", tokenizer.merges)
print('---')
print("special tokens: ", tokenizer.special_tokens)

ids = tokenizer.encode(encode_text, special_tokens="all")
print('---')
print(ids)

decode_text = tokenizer.decode(ids)
print('---')
print(decode_text)
