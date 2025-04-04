from __future__ import annotations

from dataclasses import dataclass

import av
import numpy as np
from fractions import Fraction


@dataclass
class VideoReader:
    path: str | bytes

    def __post_init__(self) -> None:
        self.container = av.open(self.path)
        self.stream = next(s for s in self.container.streams if s.type == "video")
        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height
        self.fps = float(self.stream.average_rate) if self.stream.average_rate else 30.0
        self.nframes = self.stream.frames if self.stream.frames > 0 else None

    def __iter__(self):
        for frame in self.container.decode(self.stream):
            yield frame.to_ndarray(format="rgb24")

    def close(self) -> None:
        """Close the underlying container."""

        self.container.close()


@dataclass
class VideoWriter:
    path: str
    width: int
    height: int
    fps: float
    codec: str = "libx264"
    crf: int = 18

    def __post_init__(self) -> None:
        self.container = av.open(self.path, mode="w")
        rate = Fraction(self.fps).limit_denominator(1000)
        self.stream = self.container.add_stream(self.codec, rate=rate)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(self.crf)}

    def write(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
