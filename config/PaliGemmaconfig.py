from config.Gemmaconfig import GemmaConfig
from config.SiglipVisionConfig import SiglipVisionConfig

class PaliGemmaConfig():
    """
    Configuration class for the PaliGemma model.

    This class stores the configuration parameters required to initialize the PaliGemma model, which combines vision and text configurations.

    Attributes:
        vision_config (SiglipVisionConfig): The configuration for the vision model.
        text_config (GemmaConfig): The configuration for the text model.
        ignore_index (int): The index to ignore during training. Default is -100.
        image_token_index (int): The index of the image token. Default is 256000.
        vocab_size (int): The size of the vocabulary. Default is 257152.
        projection_dim (int): The dimension of the projection layer. Default is 2048.
        hidden_size (int): The size of the hidden layers. Default is 2048.
        pad_token_id (int): The ID of the padding token. Default is None.
        is_encoder_decoder (bool): Whether the model is an encoder-decoder model. Default is False.
    """
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        """
        Initializes the PaliGemmaConfig class with the specified parameters.

        Args:
            vision_config (dict, optional): The configuration for the vision model. Default is None.
            text_config (dict, optional): The configuration for the text model. Default is None.
            ignore_index (int, optional): The index to ignore during training. Default is -100.
            image_token_index (int, optional): The index of the image token. Default is 256000.
            vocab_size (int, optional): The size of the vocabulary. Default is 257152.
            projection_dim (int, optional): The dimension of the projection layer. Default is 2048.
            hidden_size (int, optional): The size of the hidden layers. Default is 2048.
            pad_token_id (int, optional): The ID of the padding token. Default is None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim
        print("PaliGemmaConfig: ", self.vision_config.projection_dim)