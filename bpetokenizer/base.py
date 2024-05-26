"""
This file will contains all the helper functions
and Base class which has the methods to save/load model,
also required to build the BPETokenizer.
"""

def get_stats(tokens) -> dict:
    """Get statistics of the tokens. Includes the frequency of each consecutive pair of tokens"""
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx) -> list:
    """Merge the pair of tokens in the ids(list of tokens) with representing it with idx(new token in the vocab)."""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


import unicodedata

def replace_control_characters(s: str) -> str:
    """
    Replace control characters in a string with their unicode escape sequences. Prevents distortion
    Example:
        token = b"hello\nworld\x00"
        print(token) -> hello
                        world (and \x00 might not be visible)
        print(replace_control_characters(token))
        -> hello\u000aworld\u0000

    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C": # the category of the `\*` chars start with C
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


def render_token(t: bytes) -> str:
    s = t.decode('utf-8', errors='replace') # this will replace the unkown with a �
    s = replace_control_characters(s)
    return s



class Tokenizer:
    """A Base class for the tokenizer"""

    def __init__(self):
        self.merges = {}
        self.pattern = "" # the regex pattern
        self.vocab = self._build_vocab()
        self.special_tokens = {}

    def _build_vocab(self) -> dict:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_name):
        model_file = file_name + ".model"
        with open(model_file, 'w') as f:
            f.write("bpetokenizer v0.1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            for idx1, idx2 in self.merges: # this will give the tokens of pair which are merged
                f.write(f"{idx1} {idx2}\n")

            vocab_file = file_name + ".vocab"
            inverted_merges = {idx: pair for pair, idx in self.merges.items()}
            with open(vocab_file, "w", encoding="utf-8") as f:
                for idx, token in self.vocab.items():
                    s = render_token(token)
                    # find the children of this token, if any
                    if idx in inverted_merges:
                        # if this token has children, render it nicely as a merge
                        idx0, idx1 = inverted_merges[idx]
                        s0 = render_token(self.vocab[idx0])
                        s1 = render_token(self.vocab[idx1])
                        f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                    else:
                        # otherwise this is leaf token, just print it
                        # (this should just be the first 256 tokens, the bytes)
                        f.write(f"[{s}] {idx}\n")
            
    def load(self, file_name):
        assert file_name.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(file_name, 'r', encoding="utf-8") as f:
            assert f.readline().strip() == "bpetokenizer v0.1"
            self.pattern = f.readline().strip().split()
            num_special = int(f.readline().strip()) # no of lines of special_tokens
            for _ in range(num_special):
                special, idx = f.readline().strip().split()
                special_tokens[special] = int(idx)
            for line in f:
                idx1, idx2 = map(int, line.strip().split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

    def encode(self, texts):
        ...

    def decode(self, ids):
        ...

    def train(self, texts, vocab_size, verbose=False):
        ...