class SiglipVisionConfig:
    """
    Configuration class for the SiglipVision model.

    This class stores the configuration parameters required to initialize the SiglipVision model.

    Attributes:
        hidden_size (int): The size of the hidden layers. Default is 768.
        intermediate_size (int): The size of the intermediate layers. Default is 3072.
        num_hidden_layers (int): The number of hidden layers. Default is 12.
        num_attention_heads (int): The number of attention heads. Default is 12.
        num_channels (int): The number of channels in the input image. Default is 3.
        image_size (int): The size of the input image. Default is 224.
        patch_size (int): The size of the patches to divide the image into. Default is 16.
        layer_norm_eps (float): The epsilon value for layer normalization. Default is 1e-6.
        attention_dropout (float): The dropout rate for attention layers. Default is 0.0.
        num_image_tokens (int): The number of image tokens. Default is None.
    """

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        """
        Initializes the SiglipVisionConfig class with the specified parameters.

        Args:
            hidden_size (int, optional): The size of the hidden layers. Default is 768.
            intermediate_size (int, optional): The size of the intermediate layers. Default is 3072.
            num_hidden_layers (int, optional): The number of hidden layers. Default is 12.
            num_attention_heads (int, optional): The number of attention heads. Default is 12.
            num_channels (int, optional): The number of channels in the input image. Default is 3.
            image_size (int, optional): The size of the input image. Default is 224.
            patch_size (int, optional): The size of the patches to divide the image into. Default is 16.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Default is 1e-6.
            attention_dropout (float, optional): The dropout rate for attention layers. Default is 0.0.
            num_image_tokens (int, optional): The number of image tokens. Default is None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens