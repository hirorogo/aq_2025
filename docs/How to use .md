# じゃんけん画像分類AIプロジェクト - 使い方ガイド（最新版・統合版）

## 🎯 プロジェクト概要

このプロジェクトは、**シンギュライティバトルクエスト**大会用のじゃんけん画像分類AIを作成するものです。
手の画像（グー・チョキ・パー）を自動判別し、高精度を目指します。

---

## 📁 プロジェクト構成（整理後）

```
/Users/hiro/Documents/aq_2025/
├── scripts/                # 実行スクリプト
│   ├── janken_train_enhanced.py      # 強化版モデル訓練（推奨）
│   ├── janken_train_improved.py      # EfficientNetB0訓練
│   ├── janken_predict_improved.py    # 改良モデル評価
│   └── simple_predict.py             # シンプル予測
├── analysis/               # 分析スクリプト
│   └── comprehensive_analysis.py     # 包括的分析
├── docs/                   # ドキュメント
│   ├── How to use .md                # このファイル
│   ├── PROJECT_GUIDE.md              # 完全ガイド
│   ├── EXPERIMENT_LOG.md             # 実験記録
│   └── MODEL_COMPARISON_REPORT.md    # モデル比較
├── results/                # 評価・予測結果
│   ├── evaluation_result*.txt
│   ├── result*.csv
│   └── *.png
├── models/                 # 学習済みモデル
│   ├── janken_model_safe.keras
│   ├── janken_model_improved.keras
│   └── ...
├── img_train/              # 学習用画像
│   ├── 0_gu/
│   ├── 1_tyoki/
│   └── 2_pa/
├── img_test/               # テスト用画像
│   ├── 0_gu/
│   ├── 1_tyoki/
│   └── 2_pa/
├── archive/                # 古いファイル
└── venv/                   # 仮想環境
```

---

## 🚀 初期セットアップ

1. **プロジェクトディレクトリに移動**
   ```bash
   cd /Users/hiro/Documents/aq_2025
   ```
2. **仮想環境の作成・有効化**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **必要なライブラリのインストール**
   ```bash
   pip install --only-binary=all numpy matplotlib pillow scikit-learn tensorflow keras opencv-python seaborn pandas psutil
   ```
4. **SSL証明書の更新（Macのみ）**
   ```bash
   /Applications/Python\ 3.13/Install\ Certificates.command
   ```

---

## 📸 画像データの準備

- `img_train/0_gu/`, `img_train/1_tyoki/`, `img_train/2_pa/` に学習用画像を配置
- `img_test/0_gu/`, `img_test/1_tyoki/`, `img_test/2_pa/` にテスト用画像を配置
- JPG/PNG/JPEG対応、推奨224x224ピクセル

---

## 🤖 モデルの学習・評価・予測

### 1. 強化版モデルの学習（推奨）
```bash
python3 scripts/janken_train_enhanced.py
```
- Focal Loss・クラス重み付け・高度なデータ拡張
- 終了後 `models/janken_model_enhanced.keras` が生成

### 2. 改良モデルの評価
```bash
python3 scripts/janken_predict_improved.py
```
- 最新モデルでテスト画像を評価
- 結果は `results/` に保存

### 3. シンプルな予測（単一画像など）
```bash
python3 scripts/simple_predict.py
```

### 4. 包括的な分析
```bash
python3 analysis/comprehensive_analysis.py
```

---

## 📊 結果の確認

- `results/evaluation_result_*.txt`：詳細な評価レポート
- `results/result*.csv`：予測結果
- `results/*.png`：学習履歴グラフ

---

## 🛠️ トラブルシューティング

- **SSL証明書エラー**：証明書更新コマンドを再実行
- **メモリエラー**：バッチサイズを小さく、他アプリを終了
- **データ不足/偏り**：各クラス画像を追加
- **学習が進まない**：データの質・量・多様性を見直す
- **パスエラー**：ディレクトリ構成を再確認
- **詳細は `docs/DO_IT_MANUALLY.md` 参照**

---

## 🎯 よく使うコマンドまとめ

- 仮想環境有効化：
  ```bash
  source venv/bin/activate
  ```
- 学習：
  ```bash
  python3 scripts/janken_train_enhanced.py
  ```
- 評価：
  ```bash
  python3 scripts/janken_predict_improved.py
  ```
- シンプル予測：
  ```bash
  python3 scripts/simple_predict.py
  ```
- 分析：
  ```bash
  python3 analysis/comprehensive_analysis.py
  ```

---

## 📚 参考・詳細ドキュメント
- `docs/PROJECT_GUIDE.md`：全体ガイド
- `docs/EXPERIMENT_LOG.md`：実験記録
- `docs/MODEL_COMPARISON_REPORT.md`：モデル比較
- `docs/DO_IT_MANUALLY.md`：AIなし手順

---

**Good Luck! 🎉**

このガイドに従えば、最新のディレクトリ構成・スクリプトで迷わず作業できます。