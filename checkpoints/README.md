# Checkpoints Directory

This directory contains model checkpoints and training artifacts.

## Large Files

Large checkpoint files have been moved to Git LFS or GitHub Releases to keep the repository lightweight.

## Available Checkpoints

- `flow_stage1_final.pt` - Stage1 trained flow model
- `menu_stage2_final.pt` - Stage2 trained menu model

## Download Instructions

For large checkpoints, please download from:
- GitHub Releases: [Latest Release](https://github.com/KoshiHarashima/bundle-flow/releases)
- Git LFS: `git lfs pull` (if Git LFS is configured)

## Usage

```bash
# Download checkpoints to this directory
# Then run training with:
bundleflow-stage2 --cfg conf/stage2.yaml
```
