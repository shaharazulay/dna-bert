import os
import logging
import json
from typing import List
import textwrap

import pandas as pd
import torch
from torch.utils.data import Dataset


logging.basicConfig()
_logger = logging.getLogger('kmers')


class KmerTokenizer(object):
	"""
	A k-mer tokenizer, matching sequences of length k to a pre-determined k-mer index.
	:param vocab_path: str, path to the pre-"trained" k-mer token assignment.
	"""
	def __init__(self, vocab_path: str):
		with open(vocab_path, 'r') as f_vocab:
			self._vocab = json.load(f_vocab)

		self._pad_len = None

	def encode(self, kmers: List[str], add_special_tokens=True) -> List[int]:
		"""
		encode a list of kmers (k-length strings) into a list of token ids using the vocab.
		"""
		tokens = [self._vocab[kmer] for kmer in kmers]
		if add_special_tokens:
			tokens_with_special = [self._vocab['[CLS]']] + tokens + [self._vocab['[SEP]']]
		else:
			tokens_with_special = tokens

		if self._pad_len is not None:
			tokens_truncated = tokens_with_special[:self._pad_len]
			tokens_padded = tokens_truncated + [self._vocab['[PAD]']] * (self._pad_len - len(tokens_truncated))
			return tokens_padded
		return tokens_with_special

	def enable_padding(self, length):
		self._pad_len = length

	def get_vocab(self):
		return self._vocab

	def token_to_id(self, token):
		return self._vocab[token]


class BERT16SKmerDataset(Dataset):
	"""
	A torch dataset class designed to load a 16S data found in a tsv file and encode it for BERT using k-mer tokenization.
	:param vocab_path: str, path to the pre-trained bert tokenizer vocab file.
	:param data_path: str, path to the 16S data file.
	:param block_size: str, maximal BERT input (an encoded sample will be padded to this length if too short)
	:param k: int, he k-mer size to use.
	"""
	def __init__(self, vocab_path: str, data_path: str, block_size=512, k=6):

		assert os.path.isfile(data_path)
		assert os.path.isfile(vocab_path)

		_logger.info(f"Loading K-mer tokenizer using vocab file {vocab_path}")
		self.tokenizer = KmerTokenizer(vocab_path)
		self.tokenizer.enable_padding(length=block_size)

		_logger.info(f"Loading 16S dataset file at {data_path}...")
		self._16s_corpus_df = pd.read_csv(data_path, sep='\t')
		_logger.info(f"16S corpus is of shape {self._16s_corpus_df.shape}")

		self.samples = self._16s_corpus_df.seq.values.tolist()
		self.k = k

	def __len__(self):
		return len(self._16s_corpus_df)

	def __getitem__(self, i):
		sample = self.split_seq_into_kmers(self.samples[i])
		token_ids = self.tokenizer.encode(sample)
		return torch.tensor(token_ids, dtype=torch.long)

	def split_seq_into_kmers(self, seq):
		"""
		Spilt a 16S gene seqeunce into k-mers
		"""
		kmers = textwrap.wrap(seq, self.k)
		if len(kmers[-1]) < self.k:
			return kmers[:-1]
		return kmers
