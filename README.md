# BundleFlow

**Rectified Flow–based menus for combinatorial auctions**

## 🚀 Getting Started

**→ [COLAB_SETUP_GUIDE.md](COLAB_SETUP_GUIDE.md)**

## 📖 Research Background

BundleFlow is a novel approach to combinatorial auction design using Rectified Flow models. This implementation provides:

- **Stage 1**: Flow initialization for bundle generation
- **Stage 2**: Menu optimization for revenue maximization
- **Numerical stability**: Log-sum-exp, softplus constraints, warmup scheduling
- **GPU acceleration**: CUDA/MPS support with automatic optimization
- **Reproducibility**: Deterministic algorithms and comprehensive environment checks

## 📚 References

- [Rectified Flow for Economists](RECTIFIED_FLOW_FOR_ECONOMISTS.md)
- [Gumbel-Softmax Solution](GUMBEL_SOFTMAX_SOLUTION.md)
- [Technical Issues Analysis](TECHNICAL_ISSUES_ANALYSIS.md)
- [Implementation Report](IMPLEMENTATION_REPORT.md)

## 🛠️ Development

### Quick Commands

```bash
make env        # 環境構築
make test       # テスト実行
make format     # コードフォーマット
make lint       # リントチェック
make reproduce  # 5分で再現（小規模）
```

### Project Structure

```
bundle-flow/
├─ bundleflow/          # コアパッケージ
│  ├─ models/           # モデル（BundleFlow, MenuElement, Mechanism）
│  ├─ valuation/        # 評価関数（XORValuation）
│  ├─ train/            # 学習スクリプト（Stage1, Stage2）
│  ├─ data.py           # データローダー
│  └─ utils.py          # ユーティリティ
├─ src/                 # エントリポイント（後方互換性）
├─ conf/                # 設定ファイル
├─ tools/               # 環境チェック
├─ tests/               # テスト
├─ checkpoints/         # チェックポイント
├─ MODEL.md             # モデル記号と目的のドキュメント
├─ COLAB_SETUP_GUIDE.md # 唯一の入口ドキュメント
└─ pyproject.toml       # パッケージ設定
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📝 Citation

If you use this software, please cite it as described in [CITATION.cff](CITATION.cff).