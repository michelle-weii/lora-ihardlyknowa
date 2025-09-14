import os
from pathlib import Path

import torch
from torch import nn

from models.base import BasePipeline
from .wan import WanPipeline


class _AudioEncoderModule(nn.Module):
    def __init__(self, target_sample_rate: int = 16000, max_seconds: int = 10):
        super().__init__()
        # Dummy parameter ensures .to(device) works and model has parameters
        self._anchor = nn.Parameter(torch.zeros(1))
        self.target_sample_rate = target_sample_rate
        self.max_seconds = max_seconds

    def forward(self, x):
        return x


def _load_audio_waveform(file_path: str, target_sample_rate: int, max_samples: int) -> torch.Tensor:
    import av
    import numpy as np

    container = av.open(file_path)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if audio_stream is None:
        return torch.zeros(max_samples, dtype=torch.float32)

    # Resample using libswresample if needed
    audio_stream.codec_context.sample_rate = audio_stream.codec_context.sample_rate or target_sample_rate

    samples = []
    for frame in container.decode(audio_stream):
        arr = frame.to_ndarray()
        # arr shape: (channels, samples)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)  # mono
        samples.append(arr.astype(np.float32) / (np.abs(arr).max() + 1e-9))
        if sum(len(s) for s in samples) >= max_samples:
            break
    container.close()

    if len(samples) == 0:
        return torch.zeros(max_samples, dtype=torch.float32)

    wav = np.concatenate(samples, axis=0)
    # Pad / truncate to fixed length
    if len(wav) >= max_samples:
        wav = wav[:max_samples]
    else:
        pad = np.zeros(max_samples - len(wav), dtype=np.float32)
        wav = np.concatenate([wav, pad], axis=0)
    return torch.from_numpy(wav)


class WanS2VPipeline(WanPipeline):
    name = 'wan_s2v'
    # Do not treat control_file as an image to be VAE-encoded
    treat_control_as_image = False
    # Allow LoRA to target S2V attention blocks when present
    adapter_target_modules = ['WanS2VAttentionBlock', 'WanAttentionBlock']

    def __init__(self, config):
        super().__init__(config)
        # Minimal audio encoder placeholder
        self.audio_encoder = _AudioEncoderModule()

    def get_text_encoders(self):
        encoders = super().get_text_encoders()
        # Append audio encoder module so it participates in caching moves
        encoders.append(self.audio_encoder)
        return encoders

    def get_call_vae_fn(self, vae_and_clip):
        # Accept an optional control tensor argument (audio presence will set edit mode in dataset code)
        def fn(tensor, _control_tensor=None):
            # Reuse parent behavior, ignoring control
            vae = vae_and_clip.vae
            return {'latents': vae.encode(tensor.to(vae.device, vae.dtype))}
        return fn

    def get_preprocess_media_file_fn(self):
        parent_fn = super().get_preprocess_media_file_fn()
        audio_exts = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}

        def wrapper(spec, mask_filepath, size_bucket=None):
            # If this is an audio file (control side), fabricate a placeholder tensor
            path = spec[1]
            if path is not None and os.path.splitext(path)[1].lower() in audio_exts:
                # size_bucket: (w, h, frames)
                if size_bucket is None:
                    # Fallback to a minimal sane default
                    w, h, frames = 512, 512, 33
                else:
                    w, h, frames = size_bucket
                tensor = torch.zeros((3, frames, int(h), int(w)), dtype=torch.float32)
                return [(tensor, None)]
            return parent_fn(spec, mask_filepath, size_bucket)

        return wrapper

    def get_call_text_encoder_fn(self, text_encoder):
        # Text encoder path: same as WanPipeline
        if text_encoder is not self.audio_encoder:
            return super().get_call_text_encoder_fn(text_encoder)

        # Audio path: expects (caption, is_video, aux_file) where aux_file is audio path
        target_sr = self.audio_encoder.target_sample_rate
        max_samples = target_sr * self.audio_encoder.max_seconds

        def fn(caption, is_video, aux_file):
            # aux_file is a list of strings or None
            waveforms = []
            for p in aux_file or [None] * len(caption):
                if p is None or not p or not os.path.exists(p):
                    waveforms.append(torch.zeros(max_samples, dtype=torch.float32))
                else:
                    try:
                        waveforms.append(_load_audio_waveform(p, target_sr, max_samples))
                    except Exception:
                        waveforms.append(torch.zeros(max_samples, dtype=torch.float32))
            audio_waveform = torch.stack(waveforms)  # (B, T)
            return {'audio_waveform': audio_waveform}

        return fn

    # Training will ignore audio for now until transformer integration is added.
    # We intentionally do not override prepare_inputs() so existing Wan path runs.


