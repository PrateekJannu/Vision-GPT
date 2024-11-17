import torch
import torch.nn as nn

class GemmaMLP(nn.Module):
    def __init__(self, config):
        """
        Initializes the GemmaMLP module.

        Args:
            config (object): Configuration object containing the following attributes:
                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        """
        Forward pass of the GemmaMLP module.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch_Size, Seq_Len, Hidden_Size].

        Returns:
            torch.Tensor: Output tensor of shape [Batch_Size, Seq_Len, Hidden_Size].

        The forward pass consists of the following steps:
        1. Project the input tensor `x` to the intermediate size using `gate_proj`.
        2. Apply the GELU activation function to the projected tensor.
        3. Project the input tensor `x` to the intermediate size using `up_proj`.
        4. Element-wise multiply the two intermediate tensors.
        5. Project the resulting tensor back to the hidden size using `down_proj`.

        The operations can be summarized as:
        y = self.gate_proj(x)
        y = torch.gelu(y, approximate="tanh")
        j = self.up_proj(x)
        z = y * j
        z = self.down_proj(z)
        """
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

