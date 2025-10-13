# BundleFlow

**🎯 Rectified Flow–based menus for combinatorial auctions**

## 📖 包括的ガイド

**→ [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md)** - 完全なドキュメント（セットアップから高度な設定まで）

## 🚀 Quick Start

### 🎯 Demo Notebook
**→ [BundleFlow.ipynb](BundleFlow.ipynb)** - 新しいAPI構造のデモンストレーション

### ⚡ 最小実行例

```bash
# 1. インストール
pip install -e .

# 2. 基本動作確認
python -c "from bundleflow.models.flow import BundleFlow; print('✅ インストール成功')"

# 3. Stage1/2学習
bundleflow-stage1 --cfg conf/stage1.yaml
bundleflow-stage2 --cfg conf/stage2.yaml
```

## 📖 プロジェクト概要

BundleFlowは、Rectified Flowモデルを使用した組み合わせオークションのための革新的なメニュー最適化システムです。

**主な特徴:**
- **Stage 1**: Flow初期化による束生成
- **Stage 2**: メニュー最適化による収入最大化
- **新しいAPI構造**: 明確な関心の分離
- **型安全性**: 完全な型注釈と経済記号のドキュメント
- **数値安定性**: Log-sum-exp、softplus制約、ウォームアップスケジューリング
- **GPU加速**: CUDA/MPSサポートと自動最適化
- **再現性**: 決定論的アルゴリズムと包括的な環境チェック

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
├─ BundleFlow_Colab_Demo.ipynb # Colabデモノートブック
├─ COLAB_SETUP_GUIDE.md # 詳細セットアップガイド
└─ pyproject.toml       # パッケージ設定
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📝 Citation

If you use this software, please cite it as described in [CITATION.cff](CITATION.cff).