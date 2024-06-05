"""
Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

The Byte Pair Encoding (BPE) algorithm is a simple algorithm that builds a vocabulary
of subword units for a given text corpus. 

More detailed information could be found in:
https://github.com/Hk669/bpetokenizer/blob/main/notebooks/tokenization.ipynb
https://en.wikipedia.org/wiki/Byte_pair_encoding
https://youtu.be/zduSFxRajkE?si=Qv-yX2NUY69aIjCQ (Andrej Karpathy's tutorial on Tokenizer)

"""

from .base import Tokenizer, get_stats, merge
import regex as re
import os

# from the openai/tiktoken (used in gpt4 tokenizer)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""     # raw string


class BPETokenizer(Tokenizer):
    """Byte Pair Encoding tokenizer. Which handles the special tokens and the pattern for tokenization."""

    def __init__(self, pattern=None, special_tokens=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {} if special_tokens is None else special_tokens
        self.inverse_special_tokens = {} if special_tokens is None else {v: k for k, v in special_tokens.items()}


    @classmethod
    def from_pretrained(cls, 
                        tokenizer_name: str, 
                        verbose=False):
        tokenizer = cls()
        pretrained_dir = 'bpetokenizer/pretrained'
        tokenizer_file = os.path.join(pretrained_dir, tokenizer_name, f'{tokenizer_name}.json')
        if verbose:
            print(f"loading tokenizer from: {tokenizer_file}")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"tokenizer file not found: {tokenizer_file}. Please check the tokenizer name")
        tokenizer.load(tokenizer_file, mode="json")
        return tokenizer
    

    def train(self, texts, vocab_size, verbose=False, min_frequency=1) -> None:
        """
        Train the tokenizer on the given texts and vocab size. The vocab size should be greater than 256.
        params:
            texts: str (the texts required for the tokenizer to train the vocabulary.)
            vocab_size: int (the size of the vocab, gpt4 vocab size is around 100k)
            verbose: bool (to get extra visibilty and the overview of internal processes)
            min_frequency: int (the minimum frequency of the pair to be merged and added into the vocab as a new token)
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, texts) # handles the desired pattern of tokens with regex pattern

        ids = [list(tokens.encode("utf-8")) for tokens in text_chunks]      # List[List[int]]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # vocab for first 255 bytes

        # bpe algorithm
        for i in range(num_merges):
            stats = {}
            for chunk in ids:
                get_stats(chunk, stats)
            pair = max(stats, key=stats.get) # returns the highest frequency pair

            if stats[pair] < min_frequency:
                break

            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids] # merge all the max occuring pair in the each chunk in ids
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]] # concat of bytes

            if verbose:
                print(f"merging {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} frequency")

        self.merges = merges
        self.vocab = vocab


    def _encode(self, _bytes) -> list:
        """Encode the bytes into token ids(BPE algorithm)."""
        ids = list(_bytes)
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
    

    def encode_ord(self, text) -> list:
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            if chunk in self.vocab:
                ids.append(self.vocab[chunk])
            else:
                _bytes = chunk.encode("utf-8")
                chunk_ids = self._encode(_bytes)
                ids.extend(chunk_ids)
        return ids


    def encode(self, text, special_tokens="none") -> list:
        """
        Encode the text into token ids. 
        If special_tokens is set to "all", it will include the special tokens in the ids. 
        If set to "none", it will exclude the special tokens. 
        If set to "none_raise", it will raise an error if the text contains any special tokens.
        """
        special = None
        if special_tokens == "all":
            special = self.special_tokens
        elif special_tokens == "none":
            special = {}
        elif special_tokens == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        else:
            raise ValueError(f"invalid special tokens argument: {special_tokens}")
        
        if not special:
            return self.encode_ord(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        text_chunks = re.split(special_pattern, text)
        ids = []
        for chunk in text_chunks:
            if chunk in special:
                ids.append(special[chunk])
            else:
                chunkids = self._encode(chunk.encode("utf-8"))
                ids.extend(chunkids)
        return ids


    def decode(self, ids) -> str:
        part_bytes = []
        for idx in ids:
            if idx in self.vocab: #str conversion because vocab keys are strings when loaded from json
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8")) # special tokens are not encoded in vocab
            elif idx in self.merges:
                pair = self.merges[idx]
                part_bytes.append(self.vocab[pair[0]] + self.vocab[pair[1]])
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


    def _special_tokens(self, special_tokens) -> None:
        """Set the special tokens for the tokenizer. If not passed when initializing, it will be empty."""
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    

    def tokens(self, text, verbose=False) -> list:
        text_chunks = re.findall(self.compiled_pattern, text)
        
        _tokens = []
        for chunk in text_chunks:
            _bytes = chunk.encode("utf-8")
            chunk_ids = self._encode(_bytes)
            chunk_tokens = [self.vocab[idx].decode("utf-8", errors="replace") if idx in self.vocab else f"[UNK{idx}]" for idx in chunk_ids]
            _tokens.extend(chunk_tokens)
        if verbose:
            print(f"---\nlength: {len(text_chunks)}\n")
            print(f"---\ntext chunks: {text_chunks}\n")
            print(f"---\npattern: {self.pattern}\n")
        return _tokens
    