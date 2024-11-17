import torch
import torch.nn as nn
from config.Gemmaconfig import GemmaConfig
from typing import Optional
from KV_Cache.kv_cache import KVCache
from Decoder.GemmaDecoder import GemmaDecoderLayer
from Normalization.GemmaRMSNorm import GemmaRMSNorm

class GemmaModel(nn.Module):
    """
    Gemma model class.

    This class defines the architecture of the Gemma model, which includes token embeddings, 
    multiple decoder layers, and layer normalization.

    Attributes:
        config (GemmaConfig): The configuration for the Gemma model.
        padding_idx (int): The index of the padding token.
        vocab_size (int): The size of the vocabulary.
        embed_tokens (nn.Embedding): The token embedding layer.
        layers (nn.ModuleList): A list of decoder layers.
        norm (GemmaRMSNorm): The layer normalization applied to the output.
    """
    def __init__(self, config: GemmaConfig):
        """
        Initializes the GemmaModel class with the specified configuration.

        Args:
            config (GemmaConfig): The configuration for the Gemma model.
        """
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        """
        Returns the token embedding layer.

        Returns:
            nn.Embedding: The token embedding layer.
        """
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for the GemmaModel.

        This method performs the forward pass through the model, processing the input embeddings 
        through the decoder layers and applying layer normalization.

        Args:
            attention_mask (Optional[torch.Tensor]): The attention mask to be applied. Default is None.
            position_ids (Optional[torch.LongTensor]): The position IDs for the input tokens. Default is None.
            inputs_embeds (Optional[torch.FloatTensor]): The input embeddings to the model. Default is None.
                Shape: [Batch_Size, Seq_Len, Hidden_Size].
            kv_cache (Optional[KVCache]): The key-value cache for attention. Default is None.

        Returns:
            torch.FloatTensor: The output hidden states after processing through the model.
                Shape: [Batch_Size, Seq_Len, Hidden_Size].

        Example:
            >>> config = GemmaConfig(...)
            >>> model = GemmaModel(config)
            >>> inputs_embeds = torch.randn(2, 10, config.hidden_size)  # Example tensor for input embeddings
            >>> attention_mask = torch.ones(2, 1, 1, 10)  # Example tensor for attention mask
            >>> outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            >>> print(outputs.shape)  # Output: torch.Size([2, 10, config.hidden_size])
        """
        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states, kv_cache = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states