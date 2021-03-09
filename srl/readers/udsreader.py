from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, MetadataField, LabelField, SpanField

from overrides import overrides

import itertools
import json
from typing import Dict, List, Iterator
@DatasetReader.register("uds_reader")
class UDSDatasetReader(DatasetReader):
    
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}


    def _read(self, file_path: str) -> Iterator[Instance]:

        # customize this to read in stuff
        is_divider = lambda line: line.strip() == ''
        with open(file_path, 'r') as f:
            dataset = json.load(f)
            for data in dataset:
                tokens = dataset[data]["tokens"]
                predicate_index = int(dataset[data]["predicate_head_idx"])
                argument_index = int(dataset[data]["argument_head_idx"])
                label = dataset[data].get("label", None)

                yield self.text_to_instance(tokens, predicate_index, argument_index, label)

    def text_to_instance(self,
                         tokens: List[str],
                         predicate_index: int, 
                         argument_index: int,
                         label: str) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField([Token(t) for t in tokens], self._token_indexers)

        # these names to be matched in modeling code
        fields["tokens"] = tokens
        fields["label"] = LabelField(label)
        #span fields in case longer than one string, -1 to account for 1 indexing
        fields["predicate_index"] = SpanField(span_start=predicate_index-1, span_end=predicate_index-1, sequence_field=fields["tokens"])
        fields["argument_index"] = SpanField(span_start=argument_index-1, span_end=argument_index-1, sequence_field=fields["tokens"])
        #field["gid"] = MetaDataField(graph_id)
        return Instance(fields)

