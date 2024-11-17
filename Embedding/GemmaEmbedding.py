import torch
from torch import nn

class GemmaRotaryEmbedding(nn.Module):
    """
    Rotary Embedding class for the Gemma model.

    This class implements rotary positional embeddings, which are used to encode positional information in the input sequences.

    Attributes:
        dim (int): The dimension of the embeddings (typically the head dimension).
        max_position_embeddings (int): The maximum number of position embeddings. Default is 2048.
        base (int): The base value for calculating the rotary embeddings. Default is 10000.
        inv_freq (torch.Tensor): The inverse frequency tensor used for calculating the rotary embeddings.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        Initializes the GemmaRotaryEmbedding class with the specified parameters.

        Args:
            dim (int): The dimension of the embeddings (typically the head dimension).
            max_position_embeddings (int, optional): The maximum number of position embeddings. Default is 2048.
            base (int, optional): The base value for calculating the rotary embeddings. Default is 10000.
            device (torch.device, optional): The device to store the embeddings. Default is None.
        """
        super().__init__()

        self.dim = dim # it is set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        Forward pass for the GemmaRotaryEmbedding.

        This method calculates the rotary positional embeddings for the input tensor based on the provided position IDs.

        Args:
            x (torch.Tensor): The input tensor to which the rotary embeddings will be applied.
                Shape: [Batch_Size, Num_Attention_Heads, Seq_Len, Head_Size].
            position_ids (torch.Tensor): The position IDs for the input tokens.
                Shape: [Batch_Size, Seq_Len].
            seq_len (int, optional): The sequence length. Default is None.

        Returns:
            torch.Tensor: The rotary positional embeddings applied to the input tensor.
                Shape: [Batch_Size, Num_Attention_Heads, Seq_Len, Head_Size].

        Example:
            >>> embedding = GemmaRotaryEmbedding(dim=64)
            >>> x = torch.randn(2, 8, 10, 64)  # Example tensor for input
            >>> position_ids = torch.arange(10).unsqueeze(0).repeat(2, 1)  # Example position IDs
            >>> output = embedding(x, position_ids)
            >>> print(output.shape)  # Output: torch.Size([2, 8, 10, 64])
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)