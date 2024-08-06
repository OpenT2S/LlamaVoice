from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from transformers import LlamaModel, LlamaConfig, LogitsWarper, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_2_available
from transformers.modeling_outputs import ModelOutput

from llamavoice.config import Config as DefultConfig
from llamavoice.encoder import PosteriorEncoder
from llamavoice.flow import ResidualAffineCouplingBlock
from llamavoice.decoder import (
    HiFiGANGenerator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
)
from llamavoice.utils import (
    pad_unpad_sequence,
    split_hidden_states,
    get_random_segments,
    get_segments,
)
from llamavoice.model import LamaVoiceLoss


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
        self.discriminator_config = kwargs.get(
            "discriminator_config", asdict(DefultConfig.discriminator)
        )
        self.train = PretrainedConfig(**kwargs.get("train", asdict(DefultConfig.train)))
        self.loss_config = kwargs.get("loss_config", asdict(DefultConfig.loss))
        self.stop_threshold = kwargs.get("stop_threshold", 0.5)
        super().__init__(**kwargs)


class LlamaVoice(PreTrainedModel):
    def __init__(self, config: LlamaVoiceConfig):
        super().__init__(config)
        self.use_flash_attn = config.use_flash_attn

        self.gpt, self.llama_config = self._build_llama(config.gpt_config)
        self.model_dim = int(self.gpt.config.hidden_size)

        self.text_embedding = torch.nn.Embedding(config.num_text_tokens, self.model_dim)
        self.posterior_encoder = self._build_posterior_encoder(
            config.audio_encoder_config
        )
        self.text_head = nn.Linear(self.model_dim, config.num_text_tokens)
        self.dist_head = nn.Linear(self.model_dim, 2 * self.model_dim)
        self.stop_proj = nn.Linear(self.model_dim, 1)

        self.flow = self._build_flow(config.flow_config)
        self.decoder = self._build_decoder(config.decoder_config)
        self.discriminator = self._build_discriminator(config.discriminator_config)
        # Loss criterion
        self.criterion = LamaVoiceLoss(config.loss_config)
        self.config = config

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
            out_channels=self.model_dim,
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
        return ResidualAffineCouplingBlock(  # FIXME: Causal version
            in_channels=self.model_dim,
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
        self.upsample_factor = int(np.prod(config.decoder_upsample_scales))
        return HiFiGANGenerator(
            in_channels=self.model_dim,
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

    def _build_discriminator(self, config: dict):
        return HiFiGANMultiScaleMultiPeriodDiscriminator(**config)

    def dist_sampling(self, x):
        stats = self.dist_head(x)  # (b, t, c)
        m, logs = stats.split(stats.size(2) // 2, dim=2)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs

    def forward(self, batch: dict) -> Dict[str, Optional[torch.Tensor]]:
        text_token = batch["text_token"]
        text_token_len = batch["text_token_len"]
        prompt_feats = batch["prompt_feats"]
        prompt_feats_len = batch["prompt_feats_len"]
        target_feats = batch["target_feats"]
        target_feats_len = batch["target_feats_len"]
        # target_audio = batch["target_audio"]

        # forward posterior encoder
        vae_z, vae_m, vae_logs, vae_mask = self.posterior_encoder(
            target_feats, target_feats_len
        )
        prompt_z, prompt_m, prompt_logs, prompt_mask = self.posterior_encoder(
            prompt_feats, prompt_feats_len
        )
        print("--- vae_z shape", vae_z.shape)
        print("--- prompt_z shape", prompt_z.shape)

        # forward flow
        flow_z = self.flow(vae_z, vae_mask)  # (B, H, T_feats)
        print("flow_z shape ", flow_z.shape)

        # prepare llm_target

        # encode text_token
        text_embed = self.text_embedding(text_token)
        print("text embed", text_embed.shape)
        lm_input, lm_input_len = pad_unpad_sequence(
            text_embed,
            text_token_len,
            prompt_z.transpose(1, 2),
            prompt_feats_len,
            vae_z.transpose(1, 2),
            target_feats_len,
            IGNORE_ID=0,
        )  # (B, T, C), (B, )

        # run lm forward
        print("lm input", lm_input.shape)
        outputs: BaseModelOutputWithPast = self.gpt(
            inputs_embeds=lm_input, use_cache=False, output_attentions=False
        )
        print("output", outputs.last_hidden_state.shape)
        text_logits, prompt_logits, dist_logits = split_hidden_states(
            outputs.last_hidden_state,
            text_token_len,
            prompt_feats_len,
            target_feats_len,
        )
        print("text_logits", text_logits.shape)
        print("dist_logits", dist_logits.shape)
        print("prompt_logits", prompt_logits.shape)
        text_logits = self.text_head(text_logits)
        print("text_logits", text_logits.shape)
        lm_z, lm_m, lm_logs = self.dist_sampling(dist_logits)
        plm_z, plm_m, plm_logs = self.dist_sampling(prompt_logits)  # for prompt loss
        print("lm_z lm_m lm_logs", lm_z.shape, lm_m.shape, lm_logs.shape)

        # Stop token prediction
        stop = self.stop_proj(dist_logits)
        stop = torch.sigmoid(stop)

        # get random segments
        z_segments, start_idxs = get_random_segments(
            vae_z,
            target_feats_len,
            self.config.decoder_config["segment_size"],
        )
        """
        sliced_target_audio = get_segments(
            x=target_audio,
            start_idxs=start_idxs * self.upsample_factor,
            segment_size=self.config.decoder_config["segment_size"]
            * self.upsample_factor,
        )
        """

        # forward decoder with random segments
        print(z_segments.shape)
        gen_wav = self.decoder(z_segments)
        print("--- gen_wav shape", gen_wav.shape)

        """
        # calculate discriminator outputs
        p_hat = self.discriminator(gen_wav)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(sliced_target_audio)
        """
        output = ModelOutput(
            lm_m=lm_m.transpose(1, 2),
            lm_logs=lm_logs.transpose(1, 2),
            flow_z=flow_z,
            vae_m=vae_m,
            vae_logs=vae_logs,
            vae_mask=vae_mask,
            gen_wav=gen_wav,
            stop_predict=stop,
            target_feats_len=target_feats_len,
            text_logits=text_logits,
            text_targets=text_token,
            prompt_m=prompt_m,
            prompt_logs=prompt_logs,
            plm_m=plm_m.transpose(1, 2),
            plm_logs=plm_logs.transpose(1, 2),
            predicted_audio=gen_wav,
            # target_audio=sliced_target_audio,
            z_segments=z_segments,
            start_idxs=start_idxs,
        )
        return output
        """
        loss = self.criterion(loss_input)
        print(loss)
        return {"loss", loss}
        """


def test():
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
    batch_size = 2
    text_token_len = torch.randint(5, 32, (batch_size,))
    print(text_token_len)
    text_token = torch.randint(
        0, c.num_text_tokens, (batch_size, text_token_len.max())
    )  # start and end is added
    print(text_token)
    target_feats_len = torch.randint(500, 1000, (batch_size,))
    target_feats = torch.randn(
        batch_size, c.audio_encoder_config["aux_channels"], target_feats_len.max()
    )

    hop_size = int(np.prod(c.decoder_config["decoder_upsample_scales"]))
    print("--- hop size", hop_size)
    target_audio = torch.randn(batch_size, 1, target_feats_len.max() * hop_size)

    prompt_feats_len = torch.randint(100, 300, (batch_size,))
    prompt_feats = torch.randn(
        batch_size, c.audio_encoder_config["aux_channels"], prompt_feats_len.max()
    )

    batch = {
        "text_token": text_token,
        "text_token_len": text_token_len,
        "target_feats": target_feats,
        "target_feats_len": target_feats_len,
        "prompt_feats": prompt_feats,
        "prompt_feats_len": prompt_feats_len,
        "target_audio": target_audio,
    }

    for k, v in batch.items():
        print(k, v.shape)

    out = model(batch)
    print(out)


if __name__ == "__main__":
    test()
