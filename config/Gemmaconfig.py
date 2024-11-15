class GemmaConfig():
    """
    Configuration class for the Gemma model.

    This class stores the configuration parameters required to initialize the Gemma model.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        hidden_size (int): The size of the hidden layers.
        intermediate_size (int): The size of the intermediate layers.
        num_hidden_layers (int): The number of hidden layers.
        num_attention_heads (int): The number of attention heads.
        num_key_value_heads (int): The number of key-value heads.
        head_dim (int): The dimension of each attention head. Default is 256.
        max_position_embeddings (int): The maximum number of position embeddings. Default is 8192.
        rms_norm_eps (float): The epsilon value for RMS normalization. Default is 1e-6.
        rope_theta (float): The base value for rotary positional embeddings. Default is 10000.0.
        attention_bias (bool): Whether to use bias in attention layers. Default is False.
        attention_dropout (float): The dropout rate for attention layers. Default is 0.0.
        pad_token_id (int): The ID of the padding token. Default is None.
    """
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        """
        Initializes the GemmaConfig class with the specified parameters.

        Args:
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The size of the hidden layers.
            intermediate_size (int): The size of the intermediate layers.
            num_hidden_layers (int): The number of hidden layers.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key-value heads.
            head_dim (int, optional): The dimension of each attention head. Default is 256.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Default is 8192.
            rms_norm_eps (float, optional): The epsilon value for RMS normalization. Default is 1e-6.
            rope_theta (float, optional): The base value for rotary positional embeddings. Default is 10000.0.
            attention_bias (bool, optional): Whether to use bias in attention layers. Default is False.
            attention_dropout (float, optional): The dropout rate for attention layers. Default is 0.0.
            pad_token_id (int, optional): The ID of the padding token. Default is None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id