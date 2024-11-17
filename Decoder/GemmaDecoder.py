import torch
import torch.nn as nn
from typing import Optional, Tuple
from config.Gemmaconfig import GemmaConfig
from KV_Cache.kv_cache import KVCache
from MLP.GemmaMLP import GemmaMLP
from Self_Attention.GemmaAttention import GemmaAttention
from Normalization.GemmaRMSNorm import GemmaRMSNorm

class GemmaDecoderLayer(nn.Module):
    """
    Decoder layer class for the Gemma model.

    This class defines a single decoder layer for the Gemma model, which includes self-attention, 
    multi-layer perceptron (MLP), and layer normalization components.

    Attributes:
        hidden_size (int): The size of the hidden layers.
        self_attn (GemmaAttention): The self-attention mechanism.
        mlp (GemmaMLP): The multi-layer perceptron.
        input_layernorm (GemmaRMSNorm): The layer normalization applied to the input.
        post_attention_layernorm (GemmaRMSNorm): The layer normalization applied after attention.
    """
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass for the GemmaDecoderLayer.

        This method performs the forward pass through the decoder layer, applying self-attention, 
        layer normalization, and the MLP to the input hidden states.

        Args:
            hidden_states (torch.Tensor): The input hidden states to the decoder layer.
                Shape: [Batch_Size, Seq_Len, Hidden_Size].
            attention_mask (Optional[torch.Tensor]): The attention mask to be applied. Default is None.
            position_ids (Optional[torch.LongTensor]): The position IDs for the input tokens. Default is None.
            kv_cache (Optional[KVCache]): The key-value cache for attention. Default is None.

        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]: 
            The output hidden states and optionally the updated key-value cache.
                - hidden_states (torch.FloatTensor): The output hidden states.
                - kv_cache (Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]): The updated key-value cache.

        Example:
            >>> config = GemmaConfig(...)
            >>> layer = GemmaDecoderLayer(config, layer_idx=0)
            >>> hidden_states = torch.randn(2, 10, config.hidden_size)  # Example tensor for hidden states
            >>> attention_mask = torch.ones(2, 1, 1, 10)  # Example tensor for attention mask
            >>> outputs = layer(hidden_states, attention_mask)
            >>> print(outputs[0].shape)  # Output: torch.Size([2, 10, config.hidden_size])
        """
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states