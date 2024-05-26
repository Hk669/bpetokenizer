"""
Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

- which doesnt handle the special_tokens and the pattern(regex) for tokenization

The Byte Pair Encoding (BPE) algorithm is a simple algorithm that builds a vocabulary
of subword units for a given text corpus. 

More detailed information could be found in:
https://github.com/Hk669/bpetokenizer/blob/main/notebooks/tokenization.ipynb
https://en.wikipedia.org/wiki/Byte_pair_encoding
https://youtu.be/zduSFxRajkE?si=Qv-yX2NUY69aIjCQ (Andrej Karpathy's tutorial on Tokenizer)

"""

from .base import Tokenizer, get_stats, merge


class BPETokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, texts, vocab_size, verbose=False) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        tokens = texts.encode("utf-8")
        ids = list(tokens)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # vocab for first 255 bytes

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

    def encode(self, texts) -> list:
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

    def decode(self, ids) -> str:
        bytes_str = b"".join([self.vocab[idx] for idx in ids])
        texts = bytes_str.decode("utf-8", errors="replace")
        return texts

    