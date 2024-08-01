from dataclasses import dataclass, asdict


@dataclass(repr=False, eq=False)
class GPT:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 20
    use_cache: bool = False
    max_position_embeddings: int = 4096

    spk_emb_dim: int = 192
    spk_KL: bool = False
    num_audio_tokens: int = 626
    num_vq: int = 4


@dataclass(repr=False, eq=False)
class AudioEncoder:
    aux_channels: int = 512
    hidden_channels: int = 192
    posterior_encoder_kernel_size: int = 5
    posterior_encoder_layers: int = 16
    posterior_encoder_stacks: int = 1
    posterior_encoder_base_dilation: int = 1
    global_channels: int = -1
    posterior_encoder_dropout_rate: float = 0.0
    use_weight_norm_in_posterior_encoder: bool = True


@dataclass(repr=False, eq=False)
class FLOW:
    hidden_channels: int = 192
    flow_flows: int = 4
    flow_kernel_size: int = 5
    flow_base_dilation: int = 1
    flow_layers: int = 4
    global_channels: int = -1
    flow_dropout_rate: float = 0.0
    use_weight_norm_in_flow: bool = True
    use_only_mean_in_flow: bool = True


@dataclass(repr=False, eq=False)
class Decoder:
    hidden_channels: int = 192
    decoder_channels: int = 512
    global_channels: int = -1
    decoder_kernel_size: int = 7
    decoder_upsample_scales: tuple = (8, 8, 2, 2)
    decoder_upsample_kernel_sizes: tuple = (16, 16, 4, 4)
    decoder_resblock_kernel_sizes: tuple = (3, 7, 11)
    decoder_resblock_dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    use_weight_norm_in_decoder: bool = True


@dataclass(repr=False, eq=False)
class Config:
    gpt: GPT = GPT()
    audio_encoder: AudioEncoder = AudioEncoder()
    flow: FLOW = FLOW()
    decoder: Decoder = Decoder()


if __name__ == "__main__":
    print(asdict(Config.audio_encoder))
    print(asdict(Config.gpt))
    print(asdict(Config.flow))
    print(asdict(Config.decoder))
