# Verification scripts

| Script | What it checks |
|---|---|
| `dump_reference.py` | Records the upstream ACE-Step CPU float32 run that everything is compared against |
| `check_ref.py` | This port's torch reference against upstream, per diffusion step (PCC 0.99999999997) |
| `device_check.py` | Each stage on the card and the full song |
| `compare_runs.py` | A device song against the CPU reference: velocity PCC per step, waveform and log-mel |
| `vae_check.py` | The audio VAE on the card against CPU float32 and against CPU bf16 |
| `vae_conv_sweep.py`, `vae_profile.py` | Convolution configuration and where the VAE decode time goes |
| `run_on_card.sh` | Runs any of these in the TT image (`MUSIC_TT_IMAGE`) |



## Running these outside the tree they were written in

These are the scripts as they were run, inside the private working tree this port was developed in.
They are published as the record behind the numbers on the model card, and most of them need two
edits before they will run from a clone of this repo:

1. **The package name.** Six of them (`check_ref.py`, `device_check.py`, `dump_reference.py`,
   `vae_check.py`, `vae_conv_sweep.py`, `vae_profile.py`) do `from tt import acestep_dit, ...`. `tt`
   was this port's package inside the private monorepo; it is published here as **`tt_acestep`**.
2. **The `ROOT` line.** Those scripts set `ROOT = Path(__file__).resolve().parents[2]`, the monorepo
   root, and put it on `sys.path`. From a clone the repository root is `parents[1]`.

`check_ref.py` and `dump_reference.py` additionally resolve weights under `ROOT / "vendor/ACE-Step-1.5"`
and `dump_reference.py` imports the deployment's own runner; point `ACESTEP_ROOT` at an upstream
[ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) checkout and expect to replace the runner
import. `compare_runs.py` runs as published. `run_on_card.sh` uses the same monorepo layout
(`vendor/ACE-Step-1.5/checkpoints`, `experiments/tt-acestep/`). The device scripts need a p100a.
