"""WAV in, 16 kHz mono float out.

The web app records 16-bit PCM at 16 kHz; the test-set synthesizer writes
24 kHz; some tools write 32-bit float. ``decode`` accepts all three (PCM 8,
16, 24 and 32 bit, and IEEE float) and ``to_mono_16k`` brings a clip to
what the recognizer expects. Nothing here touches disk except ``write``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TARGET_RATE = 16_000

PCM = 1
IEEE_FLOAT = 3
EXTENSIBLE = 0xFFFE


@dataclass(frozen=True)
class Clip:
    samples: np.ndarray  # float32 in [-1, 1], shape (frames,) or (frames, channels)
    rate: int

    @property
    def seconds(self) -> float:
        return self.samples.shape[0] / self.rate

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]


def decode(data: bytes) -> Clip:
    """Parse a RIFF/WAVE file held in memory."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    fmt, payload = None, None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            payload = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or payload is None or len(fmt) < 16:
        raise ValueError("WAV file has no fmt or data chunk")
    tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == EXTENSIBLE and len(fmt) >= 26:
        tag = struct.unpack("<H", fmt[24:26])[0]
    if channels < 1 or rate < 8_000:
        raise ValueError(f"unsupported WAV layout: {channels} channels at {rate} Hz")
    frame = channels * bits // 8
    payload = payload[:len(payload) - len(payload) % frame]
    if tag == PCM and bits == 16:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == PCM and bits == 8:
        samples = (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == PCM and bits == 24:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = (raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16))
        ints = np.where(ints & 0x800000, ints - (1 << 24), ints)
        samples = ints.astype(np.float32) / float(1 << 23)
    elif tag == PCM and bits == 32:
        samples = np.frombuffer(payload, dtype="<i4").astype(np.float32) / float(1 << 31)
    elif tag == IEEE_FLOAT and bits == 32:
        samples = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV encoding: format {tag}, {bits} bits")
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return Clip(samples=samples, rate=rate)


def to_mono_16k(clip: Clip) -> np.ndarray:
    """Average channels and resample to 16 kHz."""
    samples = clip.samples if clip.samples.ndim == 1 else clip.samples.mean(axis=1)
    if clip.rate != TARGET_RATE:
        g = np.gcd(clip.rate, TARGET_RATE)
        samples = resample_poly(samples, TARGET_RATE // g, clip.rate // g)
    return np.clip(samples, -1.0, 1.0).astype(np.float32)


def encode(samples: np.ndarray, rate: int) -> bytes:
    """16-bit PCM mono WAV bytes."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16,
                         PCM, 1, rate, rate * 2, 2, 16, b"data", len(pcm))
    return header + pcm


def write(path: Path, samples: np.ndarray, rate: int) -> None:
    Path(path).write_bytes(encode(samples, rate))


def read(path: Path) -> Clip:
    return decode(Path(path).read_bytes())
