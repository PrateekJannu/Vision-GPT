import torch.nn as nn
import torch
from config.SiglipVisionConfig import SiglipVisionConfig
from Encoder.SiglipVisionEncoder import SiglipEncoder

class SiglipVisionTransformer(nn.Module):
    """
    SiglipVisionTransformer is a vision transformer model that processes images and outputs encoded representations.
    
    Args:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initializes the SiglipVisionTransformer model.
        
        Args:
            config (SiglipVisionConfig): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)


    def forward(self, pixel_values):
        """
        Forward pass of the SiglipVisionTransformer model.
        
        Args:
            pixel_values (torch.FloatTensor): Input tensor of shape [Batch_Size, Channels, Height, Width].
        
        Returns:
            torch.Tensor: Output tensor after encoding and layer normalization.
        """
        hidden_states = self.embeddings(pixel_values)
        encoded_layer= self.encoder(hidden_states)
        norm_layer = self.post_layernorm(encoded_layer)
        print("SiglipVisionTransformer: ",norm_layer.shape)

        return norm_layer




class SiglipVisionEmbeddings(nn.Module):
    """
    SiglipVisionEmbeddings is responsible for converting input images into patch embeddings and adding positional encodings.
    
    Args:
        config (SiglipVisionConfig): Configuration object containing model hyperparameters.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initializes the SiglipVisionEmbeddings module.
        
        Args:
            config (SiglipVisionConfig): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass of the SiglipVisionEmbeddings module.
        
        Args:
            pixel_values (torch.FloatTensor): Input tensor of shape [Batch_Size, Channels, Height, Width].
        
        Returns:
            torch.Tensor: Output tensor of shape [Batch_Size, Num_Patches, Embed_Dim] after patch embedding and positional encoding.
        """
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        print("SiglipVisionEmbeddings: ",embeddings.shape)
        return embeddings