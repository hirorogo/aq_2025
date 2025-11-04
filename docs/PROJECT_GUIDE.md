# じゃんけん画像分類プロジェクト - 完全ガイド

## プロジェクト概要
じゃんけん（グー・チョキ・パー）の手勢を自動認識する画像分類システムの開発プロジェクト

## 🎯 プロジェクトの目標
- **主要目標**: じゃんけんの手勢（グー・チョキ・パー）を高精度で分類するAIモデルを構築
- **技術目標**: 転移学習を活用し、効率的で高性能なモデルを開発
- **精度目標**: 全体精度80%以上、各クラスの再現率70%以上

## 📁 プロジェクト構造

### メインディレクトリ
```
aq_2025/
├── データセット/
│   ├── img_train/          # 訓練用画像
│   │   ├── 0_gu/          # グー画像
│   │   ├── 1_tyoki/       # チョキ画像
│   │   └── 2_pa/          # パー画像
│   └── img_test/          # テスト用画像
│       ├── 0_gu/          # グー画像（24枚）
│       ├── 1_tyoki/       # チョキ画像（19枚）
│       └── 2_pa/          # パー画像（16枚）
│
├── models/                # 保存されたモデル
│   ├── janken_model_safe.keras        # VGG16転移学習モデル
│   ├── janken_model_improved.keras    # EfficientNetB0改良モデル
│   └── training_history*.png          # 訓練履歴グラフ
│
├── scripts/               # メインスクリプト（整理後）
├── analysis/              # 分析・評価スクリプト（整理後）
├── docs/                  # ドキュメント（整理後）
└── archive/               # 古いファイル（整理後）
```

## 🚀 現在の開発状況

### ✅ 完了済み
1. **ベースラインモデル（シンプルCNN）**
   - 精度: 25.4%
   - 基本的な性能確認

2. **VGG16転移学習モデル**
   - 精度: 57.63%
   - ファイル: `models/janken_model_safe.keras`
   - 大幅な精度向上を達成

3. **EfficientNetB0改良モデル**
   - 精度: 57.63%（同等）
   - ファイル: `models/janken_model_improved.keras`
   - **重要**: パーの認識率が12.5% → 43.75%に大幅改善

### 🔄 進行中/次のステップ
1. **強化版モデル（準備完了）**
   - Focal Loss実装
   - クラス重み付け
   - 高度なデータ拡張
   - ファイル: `janken_train_enhanced.py`

## 📊 現在のモデル性能比較

| モデル | 全体精度 | グー再現率 | チョキ再現率 | パー再現率 | 主要特徴 |
|--------|----------|------------|--------------|------------|----------|
| シンプルCNN | 25.4% | - | - | - | ベースライン |
| VGG16転移学習 | 57.63% | 100% | 42.1% | 12.5% | 大幅改善、グー偏向 |
| EfficientNetB0 | 57.63% | 100% | 15.8% | 43.8% | パー認識大幅改善 |
| 強化版（未実行） | 期待値 70%+ | - | - | - | クラス重み+Focal Loss |

## 🛠️ 主要なスクリプトと使用方法

### 1. モデル訓練
```bash
# 現在の最良モデル（EfficientNetB0）の再訓練
python3 janken_train_improved.py

# 強化版モデルの訓練（推奨）
python3 janken_train_enhanced.py
```

### 2. モデル評価
```bash
# VGG16モデルの評価
python3 janken_predict_new.py

# EfficientNetB0モデルの評価
python3 janken_predict_improved.py

# 包括的な分析
python3 comprehensive_analysis.py
```

### 3. 新しい画像の予測
```bash
# シンプルな予測（単一画像）
python3 simple_predict.py
```

## 🔧 環境セットアップ

### 必要なパッケージ
```bash
# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate

# 依存パッケージのインストール
pip install tensorflow
pip install matplotlib
pip install scikit-learn
pip install numpy
pip install pillow
```

### または既存の環境
```bash
# 既存の仮想環境を使用
source venv/bin/activate
```

## 📈 次に実行すべき作業（優先順）

### 即座に実行可能
1. **強化版モデルの訓練**
   ```bash
   cd /Users/hiro/Documents/aq_2025
   source venv/bin/activate
   python3 janken_train_enhanced.py
   ```

2. **ファイル整理**
   - 古いスクリプトのアーカイブ化
   - ディレクトリ構造の最適化

### 短期目標（1-2日）
3. **モデル性能の詳細分析**
   - 混同行列の詳細分析
   - 誤分類画像の確認
   - クラス別信頼度分析

4. **データセット改善**
   - 困難な例の追加収集
   - クラス間バランスの調整

### 中期目標（1週間）
5. **アンサンブル学習の実装**
   - 複数モデルの組み合わせ
   - 投票メカニズムの構築

6. **リアルタイム予測システム**
   - Webカメラからの入力
   - ユーザーインターフェース

## 🎨 モデル改善のアプローチ

### 現在の主要問題
1. **グー偏向問題**: モデルが他のクラスもグーと誤認識
2. **クラス不均衡**: パーとチョキの認識率が低い
3. **データセット不足**: より多様な手の形・角度・背景が必要

### 解決戦略
1. **クラス重み付け**: 少数クラスに高い重みを設定
2. **Focal Loss**: 困難な例により多くの注意を向ける
3. **データ拡張強化**: より多様な変換を適用
4. **ファインチューニング**: より多くの層を訓練対象に

## 📝 実験記録

### 学習済みモデルの詳細

#### VGG16転移学習モデル
- **作成日**: 最近
- **パラメータ数**: 約15M
- **訓練時間**: 約30分
- **特徴**: 安定した学習、グー認識は完璧だがパー認識に課題

#### EfficientNetB0改良モデル
- **作成日**: 最近
- **パラメータ数**: 4,844,710
- **訓練時間**: 約45分
- **特徴**: パー認識が大幅改善、より効率的なアーキテクチャ

## 🚨 注意事項

### 実行時の注意
1. **GPU使用**: TensorFlowがGPUを認識しているか確認
2. **メモリ使用量**: 大きなバッチサイズは避ける
3. **データパス**: 相対パスが正しく設定されているか確認

### ファイル管理
1. **モデル保存**: `models/`ディレクトリに統一
2. **結果保存**: 評価結果は日付付きで保存
3. **バックアップ**: 重要なモデルは複数コピーを保持

## 🔄 次回のセッション開始手順

1. **環境確認**
   ```bash
   cd /Users/hiro/Documents/aq_2025
   source venv/bin/activate
   python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
   ```

2. **最新状況確認**
   ```bash
   ls -la models/
   cat evaluation_result_janken_model_improved.txt
   ```

3. **次の作業選択**
   - 強化版モデル訓練: `python3 janken_train_enhanced.py`
   - ファイル整理: プロジェクト構造の最適化
   - 性能分析: `python3 comprehensive_analysis.py`

## 📚 参考資料

### 技術文書
- `MODEL_COMPARISON_REPORT.md`: 詳細なモデル比較
- `evaluation_result*.txt`: 各モデルの評価結果
- `How to use.md`: 基本的な使用方法

### 学習リソース
- 転移学習: TensorFlow公式ドキュメント
- EfficientNet: 論文とTensorFlowガイド
- Focal Loss: 原論文とKeras実装例

---

**最終更新**: 2025年11月4日  
**プロジェクト状況**: 開発継続中（性能改善フェーズ）  
**次の目標**: 強化版モデルによる70%以上の精度達成
