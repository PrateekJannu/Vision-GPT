import torch.nn as nn
import torch
from config.SiglipVisionConfig import SiglipVisionConfig
from Self_Attention.SiglipAttention import SiglipAttention
from MLP.SiglipMLP import SiglipMLP

class SiglipEncoder(nn.Module):
    """
    Encoder class for the SiglipVision model.

    This class defines the encoder architecture for the SiglipVision model, which consists of multiple encoder layers.

    Attributes:
        config (SiglipVisionConfig): The configuration for the SiglipVision model.
        layers (nn.ModuleList): A list of encoder layers.
    """

    def __init__(self, config: SiglipVisionConfig):
        """
        Initializes the SiglipEncoder class with the specified configuration.

        Args:
            config (SiglipVisionConfig): The configuration for the SiglipVision model.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the SiglipEncoder.

        This method performs the forward pass through the encoder layers, processing the input embeddings.

        Args:
            inputs_embeds (torch.Tensor): The input embeddings to the encoder.
                Shape: [Batch_Size, Num_Patches, Embed_Dim].

        Returns:
            torch.Tensor: The output of the encoder after processing the input embeddings.
                Shape: [Batch_Size, Num_Patches, Embed_Dim].

        Example:
            >>> config = SiglipVisionConfig()
            >>> encoder = SiglipEncoder(config)
            >>> inputs_embeds = torch.randn(2, 196, 768)  # Example tensor for input embeddings
            >>> outputs = encoder(inputs_embeds)
            >>> print(outputs.shape)  # Output: torch.Size([2, 196, 768])
        """
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        return hidden_states
