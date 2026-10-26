"""Deterministic signal degradations for the speech test set.

Every function takes 16 kHz (or any rate) mono float32 samples and returns
samples of the same rate. They are applied to synthesized clean clips so
that the noisy, telephone and fast-speech variants of a query differ from
the clean one only in the degradation, and every variant is reproducible
from the seed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, istft, resample_poly, sosfiltfilt, stft

SPEECH_FLOOR = 1e-4


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)) + 1e-12))


def _mix(samples: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise scaled to the given signal-to-noise ratio, measured on the
    speech's RMS (the clips have little silence)."""
    noise = noise[:len(samples)]
    if len(noise) < len(samples):
        noise = np.resize(noise, len(samples))
    target = rms(samples) / (10 ** (snr_db / 20))
    scaled = noise * (target / max(rms(noise), SPEECH_FLOOR))
    return np.clip(samples + scaled, -1.0, 1.0).astype(np.float32)


def white_noise(samples: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    return _mix(samples, rng.standard_normal(len(samples)).astype(np.float32), snr_db)


def babble(samples: np.ndarray, snr_db: float, others: list[np.ndarray],
           rng: np.random.Generator) -> np.ndarray:
    """Background speech: several other clips summed at random offsets,
    which masks words the way a café or an open office does."""
    if not others:
        raise ValueError("babble needs other clips")
    bed = np.zeros(len(samples), dtype=np.float32)
    for _ in range(min(4, len(others))):
        other = others[int(rng.integers(len(others)))]
        offset = int(rng.integers(len(other)))
        rolled = np.roll(other, offset)
        bed += np.resize(rolled, len(samples)) / max(rms(other), SPEECH_FLOOR)
    return _mix(samples, bed, snr_db)


def telephone(samples: np.ndarray, rate: int) -> np.ndarray:
    """Narrowband telephone: 300–3400 Hz band-pass through an 8 kHz
    channel, then back to the original rate."""
    sos = butter(4, [300, 3400], btype="band", fs=rate, output="sos")
    narrow = sosfiltfilt(sos, samples)
    g = np.gcd(rate, 8000)
    down = resample_poly(narrow, 8000 // g, rate // g)
    up = resample_poly(down, rate // g, 8000 // g)[:len(samples)]
    return np.clip(np.resize(up, len(samples)), -1.0, 1.0).astype(np.float32)


def laptop_mic(samples: np.ndarray, rate: int, rng: np.random.Generator) -> np.ndarray:
    """A small built-in microphone: thin low end, a soft top, fan hum and
    hiss at a mild level."""
    sos = butter(2, [150, 7000], btype="band", fs=rate, output="sos")
    thin = sosfiltfilt(sos, samples).astype(np.float32)
    t = np.arange(len(samples)) / rate
    hum = 0.3 * np.sin(2 * np.pi * 120 * t) + rng.standard_normal(len(samples))
    return _mix(thin, hum.astype(np.float32), 25.0)


def reverb(samples: np.ndarray, rate: int, rng: np.random.Generator,
           rt60: float = 0.6) -> np.ndarray:
    """A room: convolution with an exponentially decaying noise impulse
    response, the simplest model of reverberation with a given RT60."""
    n = int(rate * rt60)
    decay = np.exp(-6.9 * np.arange(n) / n)  # -60 dB at rt60
    ir = rng.standard_normal(n) * decay
    ir[0] = 3.0  # direct path
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = fftconvolve(samples, ir)[:len(samples)]
    wet *= rms(samples) / max(rms(wet), SPEECH_FLOOR)
    return np.clip(wet, -1.0, 1.0).astype(np.float32)


def speed(samples: np.ndarray, factor: float, rate: int) -> np.ndarray:
    """Change speaking rate without changing pitch: a phase vocoder with
    phase-locked resynthesis. factor 1.3 makes the clip 1.3× faster."""
    if factor == 1.0:
        return samples
    nperseg, hop = 1024, 256
    _, _, spec = stft(samples, fs=rate, nperseg=nperseg, noverlap=nperseg - hop, padded=True)
    frames = spec.shape[1]
    positions = np.arange(0, frames - 1, factor)
    out = np.zeros((spec.shape[0], len(positions)), dtype=np.complex64)
    bins = np.arange(spec.shape[0])
    expected = 2 * np.pi * hop * bins / nperseg
    phase = np.angle(spec[:, 0])
    for k, pos in enumerate(positions):
        i = int(pos)
        frac = pos - i
        a, b = spec[:, i], spec[:, min(i + 1, frames - 1)]
        mag = (1 - frac) * np.abs(a) + frac * np.abs(b)
        delta = np.angle(b) - np.angle(a) - expected
        delta -= 2 * np.pi * np.round(delta / (2 * np.pi))
        out[:, k] = mag * np.exp(1j * phase)
        phase += expected + delta
    _, y = istft(out, fs=rate, nperseg=nperseg, noverlap=nperseg - hop)
    y = y[:int(len(samples) / factor)]
    return np.clip(y * (rms(samples) / max(rms(y), SPEECH_FLOOR)), -1.0, 1.0).astype(np.float32)


CONDITIONS = {
    "clean": "as synthesized",
    "noise_20db": "white noise, 20 dB SNR",
    "noise_10db": "white noise, 10 dB SNR",
    "noise_5db": "white noise, 5 dB SNR",
    "babble_10db": "background speech, 10 dB SNR",
    "telephone": "300–3400 Hz telephone band, 8 kHz channel",
    "laptop_mic": "thin built-in microphone with hum and hiss",
    "reverb": "room reverberation, RT60 0.6 s",
}


def apply(condition: str, samples: np.ndarray, rate: int, rng: np.random.Generator,
          others: list[np.ndarray] | None = None) -> np.ndarray:
    if condition == "clean":
        return samples
    if condition.startswith("noise_"):
        return white_noise(samples, float(condition[6:-2]), rng)
    if condition.startswith("babble_"):
        return babble(samples, float(condition[7:-2]), others or [], rng)
    if condition == "telephone":
        return telephone(samples, rate)
    if condition == "laptop_mic":
        return laptop_mic(samples, rate, rng)
    if condition == "reverb":
        return reverb(samples, rate, rng)
    raise ValueError(f"unknown condition {condition!r}")
