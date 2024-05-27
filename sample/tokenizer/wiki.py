from bpetokenizer import Tokenizer

text = "aaabdaaabac"
tokenizer = Tokenizer()
tokenizer.train(text, 259, verbose=True)

ids = tokenizer.encode(text)
print(ids)
print('---')

decoded_text = tokenizer.decode(ids)
print(decoded_text)

tokenizer.save("wiki")