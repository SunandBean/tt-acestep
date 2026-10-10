# SPDX-License-Identifier: Apache-2.0
"""ACE-Step 1.5 Turbo on one Tenstorrent Blackhole p100a.

The DiT decoder and the Oobleck audio VAE decoder run on the card in TTNN. The text and
lyric encoders, the sampler and the post-processing stay on the CPU inside ACE-Step's own
environment, which talks to the card through a worker process over a Unix socket.
"""
from .acestep_dit import AceStepDiT, close_device, dram_stats, open_device
from .local import LocalWorker
from .oobleck_vae import OobleckTT
from .remote import RemoteDecoder, RemoteVae, TTWorker

__all__ = ["AceStepDiT", "OobleckTT", "TTWorker", "LocalWorker", "RemoteDecoder", "RemoteVae",
           "open_device", "close_device", "dram_stats"]
__version__ = "0.1.0"
