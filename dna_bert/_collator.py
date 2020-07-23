from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollator
from tokenizers import BertWordPieceTokenizer


@dataclass
class DataCollatorForBertWordPieceTokenizer(DataCollator):
	"""
	Data collator used for language modeling fitted for the BertWordPieceTokenizer
	- collates batches of tensors, honoring their tokenizer's pad_token
	- preprocesses batches for masked language modeling
	"""

	tokenizer: BertWordPieceTokenizer
	mlm: bool = True
	mlm_probability: float = 0.15

	def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
		batch = self._tensorize_batch(examples)
		if self.mlm:
			inputs, labels = self.mask_tokens(batch)
			return {"input_ids": inputs, "masked_lm_labels": labels}
		else:
			return {"input_ids": batch, "labels": batch}

	def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
		length_of_first = examples[0].size(0)
		are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
		if are_tensors_same_length:
			return torch.stack(examples, dim=0)
		else:
			return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.token_to_id('[PAD]'))

	def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
		"""
		labels = inputs.clone()
		# sample a few tokens in each sequence for masked-LM training (with probability 0.15 in Bert)
		probability_matrix = torch.full(labels.shape, self.mlm_probability)
		special_tokens_mask = [
			self.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
		]
		probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

		if self.tokenizer.token_to_id('[PAD]') is not None:
			padding_mask = labels.eq(self.tokenizer.token_to_id('[PAD]'))
			probability_matrix.masked_fill_(padding_mask, value=0.0)

		masked_indices = torch.bernoulli(probability_matrix).bool()
		labels[~masked_indices] = -100  # We only compute loss on masked tokens

		# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
		indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
		inputs[indices_replaced] = self.tokenizer.token_to_id('[MASK]')

		# 10% of the time, we replace masked input tokens with random word
		indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
		random_words = torch.randint(len(self.tokenizer.get_vocab().values()), labels.shape, dtype=torch.long)
		inputs[indices_random] = random_words[indices_random]

		# The rest of the time (10% of the time) we keep the masked input tokens unchanged
		return inputs, labels

	def get_special_tokens_mask(self, val: str, already_has_special_tokens: bool = True):
		"""
		returns the locations of the special tokens inside the encoded sequence (so that they won't be masked)
		:param val:
		:param already_has_special_tokens:
		:return:
		"""
		cls_token_id = self.tokenizer.token_to_id('[CLS]')
		sep_token_id = self.tokenizer.token_to_id('[SEP]')
		return [token in [cls_token_id, sep_token_id] for token in val]