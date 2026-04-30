"""Export trained torch model weights to a single .npz so the Pi can run
inference with NumPy only (no torch dependency).
"""
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
PT_PATH = ROOT / "algo" / "state" / "exit_transformer" / "exit_model.pt"
NPZ_PATH = ROOT / "algo" / "state" / "exit_transformer" / "exit_model.npz"


def main():
    ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    cfg = ckpt["config"]
    norm = ckpt["norm"]

    # Collect tensors as numpy
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}

    # Save norm stats and config alongside
    arrays["_seq_mean"] = np.array(norm["seq_mean"], dtype=np.float32)
    arrays["_seq_std"] = np.array(norm["seq_std"], dtype=np.float32)
    arrays["_sca_mean"] = np.array(norm["sca_mean"], dtype=np.float32)
    arrays["_sca_std"] = np.array(norm["sca_std"], dtype=np.float32)
    arrays["_y_mean"] = np.array(norm["y_mean"], dtype=np.float32)
    arrays["_y_std"] = np.array(norm["y_std"], dtype=np.float32)
    # Config as a small string array
    import json as _json
    arrays["_config_json"] = np.array([_json.dumps(cfg)], dtype=object).astype("U")

    np.savez(NPZ_PATH, **arrays)
    print(f"Saved {NPZ_PATH}  ({NPZ_PATH.stat().st_size/1024:.1f} KB)")
    print(f"Keys: {sorted(arrays.keys())[:8]} ... ({len(arrays)} total)")


if __name__ == "__main__":
    main()
