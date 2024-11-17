import torch
import torch.nn as nn

class SiglipMLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) module for the Siglip architecture.

    This MLP consists of two fully connected layers with a GELU activation in between.
    It processes hidden states while maintaining the batch and sequence dimensions.

    Architecture:
        Input -> FC1 -> GELU -> FC2 -> Output

    Args:
        config: Configuration object containing:
            - hidden_size (int): Dimensionality of input and output features
            - intermediate_size (int): Dimensionality of inner layer

    Attributes:
        fc1 (nn.Linear): First fully connected layer (hidden_size -> intermediate_size)
        fc2 (nn.Linear): Second fully connected layer (intermediate_size -> hidden_size)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape 
                (batch_size, num_patches, hidden_size)

        Returns:
            torch.Tensor: Output tensor of shape 
                (batch_size, num_patches, hidden_size)

        Note:
            The forward pass follows these steps:
            1. Project input to intermediate size using fc1
            2. Apply GELU activation (tanh approximation)
            3. Project back to hidden size using fc2
        """
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states