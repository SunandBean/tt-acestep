#!/usr/bin/env bash
# Run an experiments/tt-acestep script on the P100a inside the TT image.
# Set MUSIC_TT_IMAGE to the tt-metal / ttnn image the port was built against; there is no portable default.
# Paths below follow the private monorepo this port was developed in -- see README.md.
#   experiments/tt-acestep/run_on_card.sh device_check.py experiments/tt-acestep/golden/d30 [--taps] [--repeat N]
#   experiments/tt-acestep/run_on_card.sh vae_check.py experiments/tt-acestep/golden/d30 [--frames N]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SCRIPT=${1:?script in experiments/tt-acestep}
GOLDEN=$(realpath "${2:?golden directory}")
shift 2
IMAGE=${MUSIC_TT_IMAGE:?set MUSIC_TT_IMAGE to the tt-metal image the port was built against}
CHECKPOINTS="$ROOT/vendor/ACE-Step-1.5/checkpoints"
CACHE="$ROOT/data/tt-cache"  # TT_METAL_CACHE: compiled kernels survive between runs
mkdir -p "$CACHE"
exec docker run --rm --ipc host --device /dev/tenstorrent \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount type=bind,src="$ROOT",dst=/work,readonly \
    --mount type=bind,src="$CHECKPOINTS/acestep-v15-turbo",dst=/checkpoint,readonly \
    --mount type=bind,src="$CHECKPOINTS/vae",dst=/vae,readonly \
    --mount type=bind,src="$GOLDEN",dst=/golden \
    --mount type=bind,src="$CACHE",dst=/cache \
    -e TT_METAL_VISIBLE_DEVICES=0 -e MESH_DEVICE=P100 -e PYTHONUNBUFFERED=1 \
    -e TT_METAL_OPERATION_TIMEOUT_SECONDS=90 \
    "$IMAGE" python "/work/experiments/tt-acestep/$SCRIPT" /golden "$@"  # absolute: the entrypoint changes the working directory
