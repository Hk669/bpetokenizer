import sys
sys.path.append('../')

from bpetokenizer import BPETokenizer

# intializing the tokenizer
tokenizer = BPETokenizer()

# load the vocab which is pretrained
tokenizer.load("sample_bpetokenizer.json", mode="json")

text = "<|startoftext|>This method? generates the tokens! which are split, before the tokenization using the pattern: default we use the gpt4 split pattern mentioned in the tiktoken.<|endoftext|>"


# this method returns a list of tokens of the text passed.
tokens = tokenizer.tokens(text, verbose=True) # if verbose, prints the text chunks and also the pattern used to split.
print('---')
print("tokens: ", tokens)

"""
tokens:  ['<|', 'st', 'ar', 't', 'oftext', '|>', 'T', 'h', 'is', ' ', 'm', 'e', 'th', 'o', 'd', '?', ' ', 'g', 'en', 'er', 'a', 't', 'e', 's', ' the', ' token',
 's', '!', ' w', 'h', 'i', 'c', 'h', ' a', 'r', 'e', ' s', 'pl', 'i', 't', ',', ' ', 'b', 'e', 'f', 'o', 'r', 'e', ' the',
 ' tokeniz', 'a', 't', 'i', 'on', ' ', 'u', 's', 'ing', ' the', ' ', 'p', 'a', 't', 't', 'er', 'n', ':', ' ', 'd', 'e', 'f', 'a', 'u', 'l', 't', ' w', 'e', ' ', 
 'u', 'se', ' the', ' ', 'g', 'p', 't', '4', ' s', 'pl', 'i', 't', ' ', 'p', 'a', 't', 't', 'er', 'n', ' ',
 'm', 'en', 't', 'i', 'on', 'e', 'd', ' ', 'in', ' the', ' t', 'i', 'k', 't', 'o', 'k', 'en', '.', '<|', 'en', 'd', 'oftext', '|>']
"""