import torch
from typing import List, Tuple

class KVCache():

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the key and value caches for a specific layer with new key and value states.

        This method manages the key-value (KV) cache for a given layer in a neural network model. If the cache for the specified layer does not exist, it initializes it with the provided key and value states. If the cache already exists, it concatenates the new key and value states to the existing cache.

        Args:
            key_states (torch.Tensor): The new key states to be added to the cache. 
                Shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim].
            value_states (torch.Tensor): The new value states to be added to the cache. 
                Shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim].
            layer_idx (int): The index of the layer for which the cache is being updated.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated key and value caches for the specified layer.
                - key_cache (torch.Tensor): The updated key cache.
                - value_cache (torch.Tensor): The updated value cache.

        Example:
            >>> key_states = torch.randn(2, 8, 10, 64)  # Example tensor for key states
            >>> value_states = torch.randn(2, 8, 10, 64)  # Example tensor for value states
            >>> layer_idx = 0  # Example layer index
            >>> key_cache, value_cache = self.update(key_states, value_states, layer_idx)
            >>> print(key_cache.shape)  # Output: torch.Size([2, 8, 10, 64])
            >>> print(value_cache.shape)  # Output: torch.Size([2, 8, 10, 64])

        Notes:
            - The method assumes that the key and value states have the same shape.
            - The concatenation is performed along the sequence length dimension (dim=-2).
        """
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]