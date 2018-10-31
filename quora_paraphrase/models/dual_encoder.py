from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
import numpy

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, SimilarityFunction, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import BooleanAccuracy


@Model.register("dual_encoder")
class DualEncoderClassifier(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 premise_encoder: Seq2VecEncoder,
                 hypothesis_encoder: Seq2VecEncoder,
                 similarity_function: SimilarityFunction,
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DualEncoderClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.premise_encoder = premise_encoder
        self.hypothesis_encoder = hypothesis_encoder
        self.similarity_function = similarity_function

        check_dimensions_match(text_field_embedder.get_output_dim(), premise_encoder.get_input_dim(),
                                "text field embedding dim", "encoder input dim")

        self.dropout = torch.nn.Dropout(dropout)
        self.metrics = {"accuracy": BooleanAccuracy()}
        self.loss = torch.nn.BCELoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        mask_premise = util.get_text_field_mask(premise)
        mask_hypothesis = util.get_text_field_mask(hypothesis)

        # embedding and encoding of the premise
        embedded_premise = self.dropout(self.text_field_embedder(premise))
        encoded_premise = self.dropout(self.premise_encoder(embedded_premise, mask_premise))

        # embedding and encoding of the hypothesis
        embedded_hypothesis = self.dropout(self.text_field_embedder(hypothesis))
        encoded_hypothesis = self.dropout(self.hypothesis_encoder(embedded_hypothesis, mask_hypothesis))

        # calculate similarity
        logits = self.similarity_function(encoded_premise, encoded_hypothesis)

        # logits = self.classifier_feedforward(similarity)
        probs = torch.sigmoid(logits)

        output_dict = {'logits': logits, "probs": probs}
        binary_probs = torch.tensor([1 if i > 0.5 else 0 for i in probs])

        if label is not None:
            loss = self.loss(probs, label.type(torch.cuda.FloatTensor))
            for metric in self.metrics.values():
                metric(binary_probs, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts indices to string labels, and adds a ``"label"`` key to the result.
        """
        predictions = output_dict["probs"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
