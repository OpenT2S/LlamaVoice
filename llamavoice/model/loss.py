from typing import List, Optional, Tuple, Union, Dict, Any
from abc import ABC, abstractmethod
import humanfriendly
import librosa
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

from transformers.modeling_outputs import ModelOutput
from transformers.configuration_utils import PretrainedConfig

from llamavoice.utils import make_pad_mask


class GeneratorAdversarialLoss(torch.nn.Module):
    """Generator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "mse",
    ):
        """Initialize GeneratorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.criterion = self._mse_loss
        else:
            self.criterion = self._hinge_loss

    def forward(
        self,
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Calcualate generator adversarial loss.

        Args:
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs..

        Returns:
            Tensor: Generator adversarial loss value.

        """
        if isinstance(outputs, (tuple, list)):
            adv_loss = 0.0
            for i, outputs_ in enumerate(outputs):
                if isinstance(outputs_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_ = outputs_[-1]
                adv_loss += self.criterion(outputs_)
            if self.average_by_discriminators:
                adv_loss /= i + 1
        else:
            adv_loss = self.criterion(outputs)

        return adv_loss

    def _mse_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _hinge_loss(self, x):
        return -x.mean()


class DiscriminatorAdversarialLoss(torch.nn.Module):
    """Discriminator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "mse",
    ):
        """Initialize DiscriminatorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(
        self,
        outputs_hat: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcualate discriminator adversarial loss.

        Args:
            outputs_hat (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from generator.
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from groundtruth.

        Returns:
            Tensor: Discriminator real loss value.
            Tensor: Discriminator fake loss value.

        """
        if isinstance(outputs, (tuple, list)):
            real_loss = 0.0
            fake_loss = 0.0
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            real_loss = self.real_criterion(outputs)
            fake_loss = self.fake_criterion(outputs_hat)

        return real_loss, fake_loss

    def _mse_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))


class FeatureMatchLoss(torch.nn.Module):
    """Feature matching loss module."""

    def __init__(
        self,
        average_by_layers: bool = True,
        average_by_discriminators: bool = True,
        include_final_outputs: bool = False,
    ):
        """Initialize FeatureMatchLoss module.

        Args:
            average_by_layers (bool): Whether to average the loss by the number
                of layers.
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            include_final_outputs (bool): Whether to include the final output of
                each discriminator for loss calculation.

        """
        super().__init__()
        self.average_by_layers = average_by_layers
        self.average_by_discriminators = average_by_discriminators
        self.include_final_outputs = include_final_outputs

    def forward(
        self,
        feats_hat: Union[List[List[torch.Tensor]], List[torch.Tensor]],
        feats: Union[List[List[torch.Tensor]], List[torch.Tensor]],
    ) -> torch.Tensor:
        """Calculate feature matching loss.

        Args:
            feats_hat (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from generator's outputs.
            feats (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from groundtruth..

        Returns:
            Tensor: Feature matching loss value.

        """
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            feat_match_loss_ = 0.0
            if not self.include_final_outputs:
                feats_hat_ = feats_hat_[:-1]
                feats_ = feats_[:-1]
            for j, (feat_hat_, feat_) in enumerate(zip(feats_hat_, feats_)):
                feat_match_loss_ += F.l1_loss(feat_hat_, feat_.detach())
            if self.average_by_layers:
                feat_match_loss_ /= j + 1
            feat_match_loss += feat_match_loss_
        if self.average_by_discriminators:
            feat_match_loss /= i + 1

        return feat_match_loss


def to_complex(c):
    if torch.is_complex(c):
        return c
    else:
        return torch.view_as_complex(c)


class AbsFeatsExtract(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Stft(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # NOTE(kamo):
        #   The default behaviour of torch.stft is compatible with librosa.stft
        #   about padding and scaling.
        #   Note that it's different from scipy.signal.stft

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None

        stft_kwargs = dict(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=window,
            normalized=self.normalized,
            onesided=self.onesided,
        )
        stft_kwargs["return_complex"] = True
        output = torch.stft(input, **stft_kwargs)
        output = torch.view_as_real(output)

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (
                torch.div(ilens - self.n_fft, self.hop_length, rounding_mode="trunc")
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def inverse(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Inverse STFT.

        Args:
            input: Tensor(batch, T, F, 2)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        input = to_complex(input)

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            datatype = input.real.dtype
            window = window_func(self.win_length, dtype=datatype, device=input.device)
        else:
            window = None

        input = input.transpose(1, 2)

        wavs = torch.functional.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
            return_complex=False,
        )

        return wavs, ilens


class LogMel(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmel_feat, ilens


class LogMelFbank(AbsFeatsExtract):
    """Conventional frontend structure for TTS.

    Stft -> amplitude-spec -> Log-Mel-Fbank
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 256,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = 80,
        fmax: Optional[int] = 7600,
        htk: bool = False,
        log_base: Optional[float] = 10.0,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.fmin = fmin
        self.fmax = fmax

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            log_base=log_base,
        )

    def output_size(self) -> int:
        return self.n_mels

    def get_parameters(self) -> Dict[str, Any]:
        """Return the parameters required by Vocoder"""
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            window=self.window,
            n_mels=self.n_mels,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # NOTE(kamo): We use different definition for log-spec between TTS and ASR
        #   TTS: log_10(abs(stft))
        #   ASR: log_e(power(stft))

        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        input_feats, _ = self.logmel(input_amp, feats_lens)
        return input_feats, feats_lens


class MelSpectrogramLoss(torch.nn.Module):
    """Mel-spectrogram loss."""

    def __init__(
        self,
        fs: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: str = "hann",
        n_mels: int = 80,
        fmin: Optional[int] = 0,
        fmax: Optional[int] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        log_base: Optional[float] = 10.0,
    ):
        """Initialize Mel-spectrogram loss.

        Args:
            fs (int): Sampling rate.
            n_fft (int): FFT points.
            hop_length (int): Hop length.
            win_length (Optional[int]): Window length.
            window (str): Window type.
            n_mels (int): Number of Mel basis.
            fmin (Optional[int]): Minimum frequency for Mel.
            fmax (Optional[int]): Maximum frequency for Mel.
            center (bool): Whether to use center window.
            normalized (bool): Whether to use normalized one.
            onesided (bool): Whether to use oneseded one.
            log_base (Optional[float]): Log base value.

        """
        super().__init__()
        self.wav_to_mel = LogMelFbank(
            fs=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            center=center,
            normalized=normalized,
            onesided=onesided,
            log_base=log_base,
        )

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        spec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated waveform tensor (B, 1, T).
            y (Tensor): Groundtruth waveform tensor (B, 1, T).
            spec (Optional[Tensor]): Groundtruth linear amplitude spectrum tensor
                (B, T, n_fft // 2 + 1).  if provided, use it instead of groundtruth
                waveform.

        Returns:
            Tensor: Mel-spectrogram loss value.

        """
        mel_hat, _ = self.wav_to_mel(y_hat.squeeze(1))
        if spec is None:
            mel, _ = self.wav_to_mel(y.squeeze(1))
        else:
            mel, _ = self.wav_to_mel.logmel(spec)
        mel_loss = F.l1_loss(mel_hat, mel)

        return mel_loss


class KLDivergenceLoss(torch.nn.Module):
    """KL divergence loss."""

    def forward(
        self,
        z_p: torch.Tensor,
        logs_q: torch.Tensor,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
        z_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate KL divergence loss.

        Args:
            z_p (Tensor): Flow hidden representation (B, H, T_feats).
            logs_q (Tensor): Posterior encoder projected scale (B, H, T_feats).
            m_p (Tensor): Expanded text encoder projected mean (B, H, T_feats).
            logs_p (Tensor): Expanded text encoder projected scale (B, H, T_feats).
            z_mask (Tensor): Mask tensor (B, 1, T_feats).

        Returns:
            Tensor: KL divergence loss.

        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()
        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        loss = kl / torch.sum(z_mask)

        return loss


class KLDivergenceLossWithoutFlow(torch.nn.Module):
    """KL divergence loss without flow."""

    def forward(
        self,
        m_q: torch.Tensor,
        logs_q: torch.Tensor,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate KL divergence loss without flow.

        Args:
            m_q (Tensor): Posterior encoder projected mean (B, H, T_feats).
            logs_q (Tensor): Posterior encoder projected scale (B, H, T_feats).
            m_p (Tensor): Expanded text encoder projected mean (B, H, T_feats).
            logs_p (Tensor): Expanded text encoder projected scale (B, H, T_feats).
        """
        posterior_norm = D.Normal(m_q, torch.exp(logs_q))
        prior_norm = D.Normal(m_p, torch.exp(logs_p))
        loss = D.kl_divergence(posterior_norm, prior_norm).mean()
        return loss


class LamaVoiceLoss(nn.Module):
    """LamaVoice loss module."""

    def __init__(self, config):
        """Construct a LamaVoiceLoss object."""
        super(LamaVoiceLoss, self).__init__()
        self.kl_loss = KLDivergenceLossWithoutFlow()
        self.flow_kl_loss = KLDivergenceLoss()
        self.bce = nn.BCELoss()
        self.mel_loss = MelSpectrogramLoss(
            **config["mel_loss_params"],
        )
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **config["generator_adv_loss_params"]
        )
        self.feat_match_loss = FeatureMatchLoss(**config["feat_match_loss_params"])
        self.config = PretrainedConfig(**config)

    def forward(self, inputs: ModelOutput) -> torch.Tensor:
        """
        ModelOutput(
            lm_m=lm_m,
            lm_logs=lm_logs,
            flow_z=flow_z,
            vae_m=vae_m,
            vae_logs=vae_logs,
            vae_mask=vae_mask,
            gen_wav=wav,

            stop_predict=stop,
            target_feats_len=target_feats_len,

            text_logits=text_logits,
            text_targets=text_token,

            prompt_m=prompt_m,
            prompt_logs=prompt_logs,
            plm_m=plm_m,
            plm_logs=plm_logs,


            predicted_audio=gen_wav,
            target_audio=sliced_target_audio,
            predicted_d_out=p_hat,
            target_d_out=p,
        )
        """
        # print all shape
        print(
            "loss input", inputs.lm_m.shape, inputs.lm_logs.shape, inputs.flow_z.shape
        )
        print(
            "loss input",
            inputs.vae_m.shape,
            inputs.vae_logs.shape,
            inputs.vae_mask.shape,
        )
        # kl-loss
        flow_kl_loss = self.flow_kl_loss(
            inputs.flow_z, inputs.vae_logs, inputs.lm_m, inputs.lm_logs, inputs.vae_mask
        )
        kl_loss = self.kl_loss(
            inputs.vae_m, inputs.vae_logs, inputs.lm_m, inputs.lm_logs
        )
        prompt_kl_loss = self.kl_loss(
            inputs.prompt_m, inputs.prompt_logs, inputs.plm_m, inputs.plm_logs
        )
        kl_loss = (
            kl_loss * self.config.kl_coeff
            + flow_kl_loss * self.config.flow_kl_coeff
            + prompt_kl_loss * self.config.prompt_kl_coeff
        )
        print("kl loss", kl_loss)
        print("--- stop_predict", inputs.stop_predict.shape)
        stop_target = torch.ones_like(inputs.stop_predict)
        mask = (
            torch.arange(
                inputs.stop_predict.size(1), device=inputs.stop_predict.device
            )[None, :]
            < inputs.target_feats_len[:, None] - 1
        )
        print("--- mask", mask, mask.shape)
        stop_target[mask] = 0.0
        stop_loss = self.bce(inputs.stop_predict, stop_target) * self.config.stop_coeff
        print("--- stop_loss", stop_loss)

        print("--- shape", inputs.text_logits.shape, inputs.text_targets.shape)
        text_loss = F.cross_entropy(
            inputs.text_logits.transpose(1, 2), inputs.text_targets
        )
        text_loss = text_loss * self.config.text_coeff
        print("--- text_loss", text_loss)

        # mel loss
        mel_loss = self.mel_loss(inputs.predicted_audio, inputs.target_audio)
        mel_loss = mel_loss * self.config.mel_coeff

        # adv loss
        adv_loss = self.generator_adv_loss(inputs.predicted_d_out)
        adv_loss = adv_loss * self.config.generator_adv_coeff
        print("--- adv_loss", adv_loss)
        feat_match_loss = self.feat_match_loss(
            inputs.predicted_d_out, inputs.target_d_out
        )
        feat_match_loss = feat_match_loss * self.config.feat_match_coeff

        print("--- feat_match_loss", feat_match_loss)

        loss = kl_loss + stop_loss + text_loss + mel_loss + adv_loss + feat_match_loss
        print(loss)
        return loss


"""
  if self.use_flow:
    kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
else:
    kl_loss = self.kl_loss(m_q, logs_q, m_p, logs_p)


posterior_encoder output: posterior_z, posterior_mean, posterior_logstd, y_mask
if self.use_flow:
    z_flow = self.flow(posterior_z, y_mask, g=g)
else:
    z_flow = None

p_z = posterior_z
p_z = self.dropout(p_z)


# get random segments
z_segments, z_start_idxs = get_random_segments(
    p_z, feats_lengths, self.segment_size
)

 output = (wav, z_start_idxs, x_mask, y_mask, common_tuple)
if self.vocoder_generator_type == "visinger2":
    output = output + (dsp_slice.sum(1),)
if self.generator_type == "visinger2":
    output = output + (predict_mel,)

common_tuple = (
posterior_z,
z_flow,
prior_mean,
prior_logstd,
posterior_mean,
posterior_logstd,
predict_lf0,
LF0 * predict_bn_mask,
predict_dur,
gt_dur,
log_probs,
)

(
_,
z_p,
m_p,
logs_p,
m_q,
logs_q,
pred_pitch,
gt_pitch,
pred_dur,
gt_dur,
log_probs,
) = outs_

if self.use_flow:
    kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
else:
    kl_loss = self.kl_loss(m_q, logs_q, m_p, logs_p)


kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

kl_distance: 

1. with flow: flow_z, vae_logs <> prior_z, prior_logstd
2. no flow: vae_z, vae_logs <> prior_z, prior_logstd
"""
