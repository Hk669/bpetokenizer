"""
This file will contains all the helper functions
and Base class which has the methods to save/load model,
also required to build the BPETokenizer.
"""

import regex as re
from .version import __version__

def get_stats(tokens, counts=None) -> dict:
    """Get statistics of the tokens. Includes the frequency of each consecutive pair of tokens"""
    counts = {} if counts is None else counts
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
    """A Base class for the tokenizer, used for training and encoding/decoding the text without special tokens."""

    def __init__(self):
        self.merges = {}
        self.pattern = "" # the regex pattern
        self.compiled_pattern = re.compile(self.pattern) if self.pattern else ""
        self.special_tokens = {}
        self.vocab = self._build_vocab() if self.merges else {}

    def _build_vocab(self) -> dict:
        """Build the vocab from the merges and special tokens. This will be used to encode/decode the tokens."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        if self.special_tokens:
            for special, idx in self.special_tokens.items():
                vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_name, mode="file"):
        """
        Writes metadata and vocabulary information to the model and vocab files.
        mode: str, default="file" | "json" to save the model and vocab in json format.
        """
        if mode == "file":
            model_file = file_name + ".model"
            with open(model_file, 'w') as f:
                f.write(f"{__version__}\n")
                f.write(f"{self.pattern}\n")
                f.write(f"{len(self.special_tokens)}\n")
                if self.special_tokens:
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
        elif mode == "json":
            import json
            data = {
                "version": __version__,
                "pattern": str(self.pattern),
                "special_tokens": self.special_tokens,
                "merges": {str(k): v for k, v in self.merges.items()},
                "vocab": {idx: render_token(token) for idx, token in self.vocab.items()}
            }
            with open(file_name + ".json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            raise ValueError("mode should be either 'file' or 'json'")
        
            
    def load(self, file_name, mode="file"):
        """
        Load the model and vocab files to the tokenizer.
        mode: str, default="file" | "json" to load the model and vocab in json format.
        """
        if mode == "file":
            assert file_name.endswith(".model")
            merges = {}
            special_tokens = {}
            idx = 256
            with open(file_name, 'r', encoding="utf-8") as f:
                assert f.readline().strip() == __version__
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

        elif mode == "json":
            assert file_name.endswith(".json")

            import json
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert data["version"] == __version__
                pattern = data["pattern"]
                pattern_regex = re.compile(r'regex.Regex\("(.+)", flags=(regex\.\w+)\)')
                match = pattern_regex.match(pattern)
                if match:
                    self.pattern = match.group(1)
                self.special_tokens = data["special_tokens"]
                self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
                merges = data["merges"]
                self.merges = {tuple(map(int, k.strip('()').split(','))): v for k, v in merges.items()}
                vocab = data["vocab"]
                self.vocab = {int(k): v.encode("utf-8") for k, v in vocab.items()}

        

    def encode(self, texts):
        """Method to encode the text to ids."""
        text_bytes = texts.encode("utf-8") # raw bytes string
        ids = list(map(int, text_bytes))
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        """Method to decode the ids to text."""
        bytes_str = b"".join([self.vocab[idx] for idx in ids])
        text = bytes_str.decode("utf-8", errors="replace")
        return text

    def train(self, texts, vocab_size, verbose=False):
        """Method for training the tokenizer."""
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        tokens = texts.encode("utf-8")
        ids = list(tokens)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # vocab for first 255 bytes

        # bpe algorithm
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get) # returns the highest frequency pair
            idx = 256 + i

            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] # concat of bytes

            if verbose:
                print(f"merging {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} frequency")

        self.merges = merges
        self.vocab = vocab
