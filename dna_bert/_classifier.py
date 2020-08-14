import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import BertModel


from _dataset import BERT16SDataset
from _kmers import BERT16SKmerDataset


class BertClassifier(nn.Module):
	"""
	Bert Model with an additional simple classification head.
	Can be used to fine tune BERT's [CLS] output for text classification tasks.
	:param path: str, path to pre-trained BERT model.
	:param num_classes: the number of output classes.
	:param freeze_bert: bool, True if BERT's weights are frozen. Set `False` to fine-tune the BERT model
	"""

	def __init__(self, path, num_classes, freeze_bert=False):

		super(BertClassifier, self).__init__()
		# specify hidden size of BERT, hidden size of the classifier, and number of output labels
		D_in, H, D_out = 256, 1000, num_classes

		# load BERT model
		self.bert = BertModel.from_pretrained(path).train()

		# declare a one-layer classifier
		self.classifier = nn.Sequential(
			nn.Linear(D_in, H),
			nn.LeakyReLU(),
			nn.Linear(H, D_out)
		)

		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False

	def forward(self, batch):
		"""
		forward pass for the joint BERT-classifier model.
		:param batch: tensor of shape (batch_size, max_length) of token ids.
		:returns logits: tensor of shape (batch_size, num_labels)
		"""
		embeddings = self.bert.embeddings(batch)

		# extract the average of the last hidden state for classification task
		embeddings_cls = embeddings.mean(dim=1)

		# feed input to classifier to compute logits
		logits = self.classifier(embeddings_cls)
		return logits


class GeneratePhylumLabels(object):
	"""
	Add a label column to the input 16S dataframe using the Phylum as labels.
	Rate Phylums are grouped to 'other' group to reduce the number of labels.
	:param data_path: str, path to the 16S data file.
	"""
	def __init__(self, data_path: str):
		assert os.path.isfile(data_path)
		self._16s_corpus_df = pd.read_csv(data_path, sep='\t')
		self._generate_labels()

	def _group_rare_phylum(self):
		self._16s_corpus_df.loc[
			self._16s_corpus_df['phylum'].value_counts()[
				self._16s_corpus_df['phylum']].values < 200, 'phylum'] = 'other'

	def _generate_labels(self):
		self._group_rare_phylum()
		label_encoder = preprocessing.LabelEncoder().fit(self._16s_corpus_df['phylum'])
		self._16s_corpus_df['label'] = label_encoder.transform(self._16s_corpus_df['phylum'])
		self._other_label = label_encoder.transform(['other'])

	@property
	def other_label(self):
		return self._other_label

	@property
	def num_classes(self):
		return self._16s_corpus_df['label'].nunique()

	def save(self, path):
		self._16s_corpus_df.to_csv(path, sep='\t')


class TrainTestSplit(object):
	"""
	Split the labeled 16S dataset into train and test.
	:param data_path: str, path to the 16S data file.
	"""
	def __init__(self, data_path: str):
		assert os.path.isfile(data_path)
		self._16s_corpus_df = pd.read_csv(data_path, sep='\t')

	def train_test_split(self, test_ratio=0.2):
		train_df, test_df = train_test_split(
			self._16s_corpus_df,
			test_size=test_ratio,
			stratify=self._16s_corpus_df.label)
		return train_df, test_df


class BERT16SDatasetForPhylaClassification(BERT16SDataset):
	"""
	A torch dataset class designed to load a 16S data found in a tsv file and encode it for BERT.
	:param vocab_path: str, path to the pre-trained bert tokenizer vocab file.
	:param data_path: str, path to the 16S data file.
	:param block_size: str, maximal BERT input (an encoded sample will be padded to this length if too short)
	:param max_word_length: int, the maximal word length the tokenizer can encode.
	"""
	def __init__(self, vocab_path: str, data_path: str, block_size=512, max_word_length=100):
		super(BERT16SDatasetForPhylaClassification, self).__init__(vocab_path, data_path, block_size=block_size, max_word_length=max_word_length)
		self.labels = self._16s_corpus_df['label'].values.tolist()

	def __getitem__(self, i):
		input_ids = super(BERT16SDatasetForPhylaClassification, self).__getitem__(i)
		label = torch.tensor([self.labels[i]])
		return input_ids, label


class BERT16SKmerDatasetForPhylaClassification(BERT16SKmerDataset):
	"""
	A torch dataset class designed to load a 16S data found in a tsv file and encode it for BERT using k-mer tokenization.
	:param vocab_path: str, path to the pre-trained bert tokenizer vocab file.
	:param data_path: str, path to the 16S data file.
	:param block_size: str, maximal BERT input (an encoded sample will be padded to this length if too short)
	:param k: int, he k-mer size to use.
	"""
	def __init__(self, vocab_path: str, data_path: str, block_size=512, k=6):
		super(BERT16SKmerDatasetForPhylaClassification, self).__init__(vocab_path, data_path, block_size=block_size, k=k)
		self.labels = self._16s_corpus_df['label'].values.tolist()

	def __getitem__(self, i):
		input_ids = super(BERT16SKmerDatasetForPhylaClassification, self).__getitem__(i)
		label = torch.tensor([self.labels[i]])
		return input_ids, label
