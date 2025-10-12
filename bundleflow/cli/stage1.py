import torch
import os
import json
from omegaconf import OmegaConf
from bundleflow.flow import FlowModel  # 既存を利用
from bundleflow.utils import seed_all

def pick_device(s: str):
    if s != "auto": 
        return torch.device(s)
    if torch.cuda.is_available(): 
        return torch.device("cuda")
    if torch.backends.mps.is_available(): 
        return torch.device("mps")
    return torch.device("cpu")

def main(cfg_path: str | None = None):
    # 設定ロード（cfg_path 未指定なら conf/stage1.yaml）
    cfg = OmegaConf.load(cfg_path or os.path.join("conf", "stage1.yaml"))
    device = pick_device(cfg.device)
    print(f"[Stage1] device={device} torch={torch.__version__}")
    seed_all(cfg.seed, deterministic_cudnn=True)

    # ここで FlowModel を構築/学習（既存の train_stage1 の処理を関数化して呼ぶ）
    from bundleflow.train_stage1_impl import run  # 既存ロジックをここへ切出し
    run(cfg, device)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=None, help="path to YAML")
    args = ap.parse_args()
    main(args.cfg)
