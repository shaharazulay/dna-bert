import os
import logging

import torch
import pandas as pd
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset


logging.basicConfig()
_logger = logging.getLogger('bert_dataset')


class BERT16SDataset(Dataset):
	"""
	A torch dataset class designed to load a 16S data found in a tsv file and encode it for BERT.
	:param vocab_path: str, path to the pre-trained bert tokenizer vocab file.
	:param data_path: str, path to the 16S data file.
	:param block_size: str, maximal BERT input (an encoded sample will be padded to this length if too short)
	:param max_word_length: int, the maximal word length the tokenizer can encode.
	"""
	def __init__(self, vocab_path: str, data_path: str, block_size=512, max_word_length=100):

		assert os.path.isfile(data_path)
		assert os.path.isfile(vocab_path)

		_logger.info(f"Loading BERT tokenizer using vocab file {vocab_path}")
		self.tokenizer = BertWordPieceTokenizer(
			vocab_path,
			handle_chinese_chars=False,
			lowercase=False)
		self.tokenizer.enable_truncation(block_size)
		self.tokenizer.enable_padding(max_length=block_size)

		_logger.info(f"Loading 16S dataset file at {data_path}...")
		self._16s_corpus_df = pd.read_csv(data_path, sep='\t')
		_logger.info(f"16S corpus is of shape {self._16s_corpus_df.shape}")

		self.samples = self._16s_corpus_df.seq.values.tolist()
		self.max_word_length = max_word_length

	def __len__(self):
		return len(self._16s_corpus_df)

	def __getitem__(self, i):
		sample = self._split_sequence_by_max_word_length(self.samples[i])
		tokens = self.tokenizer.encode(sample)
		return torch.tensor(tokens.ids, dtype=torch.long)

	def _split_sequence_by_max_word_length(self, seq):
		"""
		split a 16S sequence (~1K long usually) into white-spaces separated chunks that the tokenizer can encode.
		:param seq: str, 16S sequence
		:return: str
		"""
		chunks = [seq[i: i + self.max_word_length] for i in range(0, len(seq), self.max_word_length)]
		return ' '.join(chunks)

