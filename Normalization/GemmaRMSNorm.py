import torch
import torch.nn as nn

class GemmaRMSNorm(nn.Module):
    """Root Mean Square (RMS) Normalization layer specific to Gemma models.
    
    This implementation follows Gemma's specific approach to RMS normalization,
    where the weight application and dtype casting order differs from other
    implementations like LLaMA.
    
    Args:
        dim (int): The dimension of the input tensor to normalize
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-6
    
    Attributes:
        eps (float): Small constant for numerical stability
        weight (nn.Parameter): Learnable scale parameter for the normalization
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """Internal method to compute RMS normalization.
        
        Args:
            x (torch.Tensor): Input tensor to normalize
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Forward pass for RMS normalization.
        
        Performs RMS normalization with learned scale parameters.
        Note: Unlike LLaMA which does x.to(float16) * w, Gemma does (x * w).to(float16)
        
        Args:
            x (torch.Tensor): Input tensor to normalize
            
        Returns:
            torch.Tensor: Normalized and scaled tensor in the same dtype as input
        """
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)