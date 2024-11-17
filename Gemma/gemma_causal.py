import torch
import torch.nn as nn

from KV_Cache.kv_cache import KVCache
from typing import Optional, Tuple
from Gemma.gemma import GemmaModel

class GemmaForCausalLM(nn.Module):
    """
    GemmaForCausalLM is a class for a causal language model using the Gemma architecture.

    Attributes:
        config (object): Configuration object containing model parameters.
        model (GemmaModel): The underlying Gemma model.
        vocab_size (int): The size of the vocabulary.
        lm_head (nn.Linear): Linear layer mapping hidden states to vocabulary logits.

    Methods:
        get_input_embeddings():
            Returns the input embeddings from the underlying Gemma model.

        tie_weights():
            Ties the weights of the lm_head to the input embeddings.

        forward(attention_mask=None, position_ids=None):
            Defines the forward pass of the model.
            Args:
                attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding token indices.
                position_ids (Optional[torch.LongTensor]): Indices of positions of each input sequence tokens in the batch.
            Returns:
                torch.Tensor: The logits of the language model.
    """
    def __init__(self, config):
        """
        Initializes the GemmaForCausalLM model with the given configuration.

        Args:
            config (object): Configuration object containing model parameters.
        """
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        """
        Returns the input embeddings from the underlying Gemma model.

        Returns:
            nn.Embedding: The input embeddings.
        """
        return self.model.embed_tokens
    
    def tie_weights(self):
        """
        Ties the weights of the lm_head to the input embeddings.
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Defines the forward pass of the model.

        Args:
            attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding token indices.
            position_ids (Optional[torch.LongTensor]): Indices of positions of each input sequence tokens in the batch.

        Returns:
            torch.Tensor: The logits of the language model.
        """
        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data