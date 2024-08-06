from dataclasses import dataclass, asdict, field
from typing import List, Dict


@dataclass(repr=False, eq=False)
class GPT:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 20
    use_cache: bool = False
    max_position_embeddings: int = 4096


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
    segment_size: int = 64


@dataclass(repr=False, eq=False)
class Discriminator:
    scales: int = 1
    scale_downsample_pooling: str = "AvgPool1d"
    scale_downsample_pooling_params: Dict[str, int] = field(
        default_factory=lambda: {
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        }
    )
    scale_discriminator_params: Dict[str, object] = field(
        default_factory=lambda: {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        }
    )
    follow_official_norm: bool = False
    periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    period_discriminator_params: Dict[str, object] = field(
        default_factory=lambda: {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        }
    )


@dataclass(repr=False, eq=False)
class Loss:
    kl_coeff: float = 1.0
    flow_kl_coeff: float = 1.0
    prompt_kl_coeff: float = 1.0
    stop_coeff: float = 1.0
    text_coeff: float = 1.0
    mel_loss_params: Dict[str, object] = field(
        default_factory=lambda: {
            "fs": 24000,  # must be the same as the training data
            "n_fft": 1024,  # fft points
            "hop_length": 256,  # hop size
            "win_length": None,  # window length
            "window": "hann",  # window type
            "n_mels": 80,  # number of Mel basis
            "fmin": 0,  # minimum frequency for Mel basis
            "fmax": None,  # maximum frequency for Mel basis
            "log_base": None,  # null represent natural log
        }
    )
    mel_coeff: float = 1.0
    generator_adv_loss_params: Dict[str, object] = field(
        default_factory=lambda: {"loss_type": "mse", "average_by_discriminators": False}
    )
    generator_adv_coeff: float = 1.0
    discriminator_adv_loss_params: Dict[str, object] = field(
        default_factory=lambda: {"loss_type": "mse", "average_by_discriminators": False}
    )
    discriminator_adv_coeff: float = 1.0
    feat_match_loss_params: Dict[str, bool] = field(
        default_factory=lambda: {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        }
    )
    feat_match_coeff: float = 1.0


@dataclass(repr=False, eq=False)
class Train:
    gradient_accumulation_step: int = 1
    tracker: tuple = ("tensorboard",)
    max_epoch: int = 1000
    save_checkpoint_stride: tuple = (500,)
    keep_last: tuple = (1,)
    run_eval: tuple = (True,)
    random_seed: int = 1024
    dataloader: Dict[str, object] = field(
        default_factory=lambda: {
            "num_worker": 4,
            "pin_memory": True,
            "persistent_workers": True,
        }
    )


@dataclass(repr=False, eq=False)
class Config:
    gpt: GPT = GPT()
    audio_encoder: AudioEncoder = AudioEncoder()
    flow: FLOW = FLOW()
    decoder: Decoder = Decoder()
    loss: Loss = Loss()
    discriminator: Discriminator = Discriminator()
    train: Train = Train()


if __name__ == "__main__":
    print(asdict(Config.audio_encoder))
    print(asdict(Config.gpt))
    print(asdict(Config.flow))
    print(asdict(Config.decoder))
    print(asdict(Config.loss))
    print(asdict(Config.discriminator))
