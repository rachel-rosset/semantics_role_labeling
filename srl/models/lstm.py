from allennlp.models import Model
import torch
import torch.nn as nn

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import F1Measure
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from typing import Dict, Optional


@Model.register('srl_lstm')
class SRLLSTM(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = torch.nn.Linear(in_features=2*encoder.get_output_dim(),
                                     out_features=vocab.get_vocab_size('labels'))

        #self._ffnn = torch.nn.Sequential(
                ##linear
                ##dropout
                ##tanh
                ##linear
            #)
        # define f1 here, use as plain F1 measure not spanBased
        self._metric = F1Measure(positive_label=vocab.get_token_index(token='positive', namespace='labels'))

        # define pytorch loss here BCEWithLogitsLoss?

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return self._metric.get_metric(reset)

    # no model params or anything specified here, all in config file
    def forward(self, 
                tokens: Dict[str, torch.Tensor],
                predicate_index: torch.Tensor,
                argument_index: torch.Tensor,
                label: Optional[torch.Tensor] = None)-> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens)

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)
        
        pred_encode = encoded[torch.arange(encoded.size(0)), torch.tensor(predicate_index[:,0])]
        arg_encode = encoded[torch.arange(encoded.size(0)), torch.tensor(argument_index[:,0])]


        concat_encode = torch.cat((arg_encode, pred_encode), 1)
        predictions = self._classifier(concat_encode)
        # use F1Measure as self.metric
        output: Dict[str, torch.Tensor] = {}
        output['logits'] = predictions
        one_zero_preds = []
        for i,j in predictions:
            if i > j:
                i = 1
                j = 0
            else:
                i = 0
                j = 1
            one_zero_preds.append(torch.tensor([i,j]))
            print("i, j", i, j)
        compare = torch.stack(one_zero_preds, 1)
        predictions = torch.flip(compare, [1])
        #d_predictions = predictions.argmax(1)

        if label is not None:
            # allennlp requires output as dictionary with loss as a key
            #probably a different loss here
            print("label shape", label)
            print("pred shape", predictions.size())
            self._metric(compare, label)
            output["loss"] = torch.nn.CrossEntropyLoss(compare, label)

        return output





