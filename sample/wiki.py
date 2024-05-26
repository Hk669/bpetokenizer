import sys
sys.path.append("../")
from bpetokenizer import BPETokenizer

text = "aaabdaaabac"
tokenizer = BPETokenizer()
tokenizer.train(text, 259, verbose=True)

ids = tokenizer.encode(text)
print(ids)
print('---')

decoded_text = tokenizer.decode(ids)
print(decoded_text)

tokenizer.save("wiki")