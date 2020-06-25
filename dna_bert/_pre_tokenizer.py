from tokenizers.pre_tokenizers import PreTokenizer

from typing import Optional, List, Tuple
Offsets = Tuple[int, int]


class DNAPreTokenizer(object):

	def __init__(self) -> None:
		super(DNAPreTokenizer).__init__()

	def pre_tokenize(self, sequence: str) -> List[Tuple[str, Offsets]]:
		return [(sequence, (0, len(sequence)))]