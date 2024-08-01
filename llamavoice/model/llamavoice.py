from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

import torch
from torch import nn

from transformers import LlamaModel, LlamaConfig, LogitsWarper, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_2_available

from llamavoice.config import Config as DefultConfig
from llamavoice.encoder import PosteriorEncoder
from llamavoice.flow import ResidualAffineCouplingBlock
from llamavoice.decoder import HiFiGANGenerator


class LlamaVoiceConfig(PretrainedConfig):
    model_type = "llamavoice"

    def __init__(self, **kwargs):
        self.use_flash_attn = kwargs.get("use_flash_attn", True)
        self.gpt_config = kwargs.get("gpt_config", asdict(DefultConfig.gpt))
        self.num_text_tokens = kwargs.get("num_text_tokens", 256)
        self.audio_encoder_config = kwargs.get(
            "audio_encoder_config", asdict(DefultConfig.audio_encoder)
        )
        self.flow_config = kwargs.get("flow_config", asdict(DefultConfig.flow))
        self.decoder_config = kwargs.get("decoder_config", asdict(DefultConfig.decoder))
        super().__init__(**kwargs)


class LlamaVoice(PreTrainedModel):
    def __init__(self, config: LlamaVoiceConfig):
        super().__init__(config)
        self.use_flash_attn = config.use_flash_attn

        self.gpt, self.llama_config = self._build_llama(config.gpt_config)
        model_dim = int(self.gpt.config.hidden_size)

        self.text_embedding = torch.nn.Embedding(config.num_text_tokens, model_dim)
        self.posterior_encoder = self._build_posterior_encoder(
            config.audio_encoder_config
        )
        self.flow = self._build_flow(config.flow_config)
        self.decoder = self._build_decoder(config.decoder_config)

    def _build_llama(
        self,
        config: dict,
    ) -> Tuple[LlamaModel, LlamaConfig]:

        if self.use_flash_attn and is_flash_attn_2_available():
            llama_config = LlamaConfig(
                **config,
                attn_implementation="flash_attention_2",
            )
            self.logger.warning(
                "enabling flash_attention_2 may make gpt be even slower"
            )
        else:
            llama_config = LlamaConfig(**config)

        model = LlamaModel(llama_config)
        del model.embed_tokens

        return model, llama_config

    def _build_posterior_encoder(self, config: dict):
        config = PretrainedConfig(**config)
        return PosteriorEncoder(
            in_channels=config.aux_channels,
            out_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            kernel_size=config.posterior_encoder_kernel_size,
            layers=config.posterior_encoder_layers,
            stacks=config.posterior_encoder_stacks,
            base_dilation=config.posterior_encoder_base_dilation,
            global_channels=config.global_channels,
            dropout_rate=config.posterior_encoder_dropout_rate,
            use_weight_norm=config.use_weight_norm_in_posterior_encoder,
        )

    def _build_flow(self, config: dict):
        config = PretrainedConfig(**config)
        return ResidualAffineCouplingBlock(
            in_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            flows=config.flow_flows,
            kernel_size=config.flow_kernel_size,
            base_dilation=config.flow_base_dilation,
            layers=config.flow_layers,
            global_channels=config.global_channels,
            dropout_rate=config.flow_dropout_rate,
            use_weight_norm=config.use_weight_norm_in_flow,
            use_only_mean=config.use_only_mean_in_flow,
        )

    def _build_decoder(self, config: dict):
        config = PretrainedConfig(**config)
        return HiFiGANGenerator(
            in_channels=config.hidden_channels,
            out_channels=1,
            channels=config.decoder_channels,
            global_channels=config.global_channels,
            kernel_size=config.decoder_kernel_size,
            upsample_scales=config.decoder_upsample_scales,
            upsample_kernel_sizes=config.decoder_upsample_kernel_sizes,
            resblock_kernel_sizes=config.decoder_resblock_kernel_sizes,
            resblock_dilations=config.decoder_resblock_dilations,
            use_weight_norm=config.use_weight_norm_in_decoder,
        )

    def forward(self, batch: dict) -> Dict[str, Optional[torch.Tensor]]:
        text_token = batch["text_token"]
        text_token_len = batch["text_token_len"]
        feats = batch["feats"]  # Feature tensor (B, aux_channels, T_feats).
        feats_len = batch["feats_len"]  # Feature length tensor (B,).

        # forward posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(feats, feats_lengths)
        # forward flow
        z_p = self.flow(z, y_mask, g=g)  # (B, H, T_feats)

        # prepare llm_target

        # encode text_token
        text_token = self.text_embedding(text_token)

        outputs: BaseModelOutputWithPast = self.gpt(
            attention_mask=model_input.attention_mask,
            position_ids=model_input.position_ids,
            past_key_values=model_input.past_key_values,
            inputs_embeds=model_input.inputs_embeds,
            use_cache=model_input.use_cache,
            output_attentions=return_attn,
            cache_position=model_input.cache_position,
        )
        del_all(model_input)

        # get random segments
        z_segments, z_start_idxs = get_random_segments(
            z,
            feats_lengths,
            self.segment_size,
        )

        # forward decoder with random segments
        wav = self.decoder(z_segments, g=g)


if __name__ == "__main__":
    c = LlamaVoiceConfig()
    print(c)
    model = LlamaVoice(c)
    print(model.device)
    print(model.gpt)
    print(model.posterior_encoder)
    print(model.flow)
    print("with weight norm", model.decoder)
    model.decoder.remove_weight_norm()
    print("without weight norm", model.decoder)

    # generate test data
    text_token = torch.randint(0, c.num_text_tokens, (2, 32))
    print(text_token)
