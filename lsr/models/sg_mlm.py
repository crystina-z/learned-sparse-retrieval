import logging

from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep
from transformers import AutoModelForMaskedLM
from pprint import pprint

import torch
from torch import nn
from transformers import PretrainedConfig


from transformers import (
    AutoModelForMaskedLM,
    BertForMaskedLM,
    BertPreTrainedModel,
)
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertOnlyMLMHead,
    BertLMPredictionHead,
    BertPredictionHeadTransform,
)

logger = logging.getLogger(__name__)


from lsr.models.mlm import (
    EPICTermImportance,
    EPICDocQuality, 
    TransformerMLMConfig,
    TransformerMLMSparseEncoder,
)


class SGOutputEmbeddingsBertLMPredictionHead(BertLMPredictionHead):
    _tied_weights_keys = None # <-- NO TIE WEIGHTS

    def __init__(self, config):
        nn.Module.__init__(self)
        self.transform = BertPredictionHeadTransform(config)

        output_vocab_size= config.output_vocab_size

        # self.decoder = nn.Linear(config.hidden_size, output_vocab_size, bias=False)
        self.decoder = nn.Linear(config.hidden_size, output_vocab_size, bias=True)
        # self.bias = nn.Parameter(torch.zeros(output_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        # self.decoder.bias = self.bias

    def _tie_weights(self):
        # self.decoder.bias = self.bias
        pass

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SGOutputEmbeddingsBertOnlyMLMHead(BertOnlyMLMHead):
    _tied_weights_keys = None # <-- NO TIE WEIGHTS

    def __init__(self, config):
        nn.Module.__init__(self)
        self.predictions = SGOutputEmbeddingsBertLMPredictionHead(config)


class BertSGOutputEmbeddingsForMaskedLM(BertForMaskedLM):
    _tied_weights_keys = None # <-- NO TIE WEIGHTS

    def __init__(self, config):
        BertPreTrainedModel.__init__(self, config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = SGOutputEmbeddingsBertOnlyMLMHead(config)

        self.post_init()


# main class for LSR
class SGOutputTransformerMLMConfig(TransformerMLMConfig):
    """
    Configuration for the SGOutputTransformerMLMSparseEncoder
    """

    model_type = "SG_MLM"


class SGOutputTransformerMLMSparseEncoder(TransformerMLMSparseEncoder):
    """
    SGOutputTransformerMLMSparseEncoder is a version of TransformerMLMSparseEncoder
    that uses compressed output word embedding 
    """

    config_class = SGOutputTransformerMLMConfig

    def __init__(self, config: SGOutputTransformerMLMConfig = SGOutputTransformerMLMConfig()):
        super(SparseEncoder, self).__init__(config)
        assert "xlm" not in config.tf_base_model_name_or_dir, "XLM is not supported in this version"
        pprint(config)

        self.model = BertSGOutputEmbeddingsForMaskedLM.from_pretrained(
            config.tf_base_model_name_or_dir
        )
        self.activation = FunctionalFactory.get(config.activation)
        self.pool = PoolingFactory.get(config.pool)
        if config.term_importance == "no":
            self.term_importance = functional.AllOne()
        elif config.term_importance == "epic":
            self.term_importance = EPICTermImportance()
        if config.doc_quality == "no":
            self.doc_quality = functional.AllOne()
        elif config.doc_quality == "epic":
            self.doc_quality = EPICDocQuality()

        self.norm = FunctionalFactory.get(config.norm)
