# じゃんけん画像分類システム - モデル性能比較レポート

## 概要
本レポートは、じゃんけん（グー・チョキ・パー）画像分類システムにおける異なるアプローチの性能を比較したものです。

## テストされたモデル

### 1. シンプルCNN（ベースライン）
- **ファイル**: `models/janken_simple_model.h5`
- **アーキテクチャ**: カスタムCNN
- **精度**: 25.4%
- **特徴**: 軽量だが性能が低い

### 2. VGG16転移学習モデル
- **ファイル**: `models/janken_model_safe.keras`
- **アーキテクチャ**: VGG16 + カスタムヘッド
- **精度**: 57.63%
- **特徴**: 
  - ベースラインから大幅な改善（+32.23%）
  - しかし、クラス間の不均衡な認識（グー: 100%, チョキ: 42.1%, パー: 12.5%）

### 3. EfficientNetB0改良モデル（開発中）
- **ファイル**: `models/janken_model_improved.keras`（訓練中）
- **アーキテクチャ**: EfficientNetB0 + 改良されたヘッド
- **期待される改善点**:
  - より軽量で効率的なアーキテクチャ
  - ファインチューニングによる精度向上
  - 強化されたデータ拡張

## 詳細分析：VGG16モデル（現在の最良）

### クラス別性能
| クラス | 適合率 | 再現率 | F値 | サンプル数 |
|--------|--------|--------|-----|-----------|
| 0_gu (グー) | 0.5217 | 1.0000 | 0.6857 | 24 |
| 1_tyoki (チョキ) | 0.7273 | 0.4211 | 0.5333 | 19 |
| 2_pa (パー) | 1.0000 | 0.1250 | 0.2222 | 16 |

### 混同行列分析
```
実際\予測    0_gu    1_tyoki    2_pa
0_gu           24         0        0
1_tyoki        11         8        0  
2_pa           11         3        2
```

### 主な問題点
1. **過度なグー偏向**: 他のクラスもグーと誤認識する傾向
2. **パー認識の困難**: パーの認識率が極めて低い（12.5%）
3. **クラス不均衡**: グーが他のクラスより多く予測される

## 改善戦略

### 実装済み改善（EfficientNetB0モデル）
1. **アーキテクチャの変更**:
   - VGG16 → EfficientNetB0（より効率的）
   - BatchNormalizationの追加
   - より適切なDropout配置

2. **データ拡張の強化**:
   - brightness_range=[0.8, 1.2]
   - channel_shift_range=20.0
   - より多様な変換パラメータ

3. **ファインチューニング**:
   - 最後の20層を訓練可能に設定
   - より柔軟な特徴学習

4. **訓練改善**:
   - 適応的学習率（ReduceLROnPlateau）
   - EarlyStoppingの最適化
   - ModelCheckpointでベストモデル保存

### 今後の改善提案
1. **データセットの改善**:
   - より多様な手の形、角度、背景
   - クラス間のバランス調整
   - 困難な例（曖昧な手の形）の追加

2. **クラス重み付け**:
   - 少数クラス（パー）に高い重みを設定
   - 損失関数の調整

3. **アンサンブル学習**:
   - 複数モデルの組み合わせ
   - 投票メカニズムの実装

## 使用方法

### 現在のベストモデルでのテスト
```bash
python3 janken_predict_new.py
```

### 改良モデルでのテスト（訓練完了後）
```bash
python3 janken_predict_improved.py
```

### 新しいモデルの訓練
```bash
python3 janken_train_improved.py
```

## ファイル構成
```
models/
├── janken_simple_model.h5          # ベースラインモデル
├── janken_model_safe.keras         # VGG16転移学習モデル
├── janken_model_improved.keras     # EfficientNetB0改良モデル
├── training_history.png            # VGG16訓練履歴
└── training_history_improved.png   # EfficientNetB0訓練履歴

predict scripts/
├── simple_predict.py               # シンプルモデル用
├── janken_predict_new.py          # VGG16モデル用
└── janken_predict_improved.py     # EfficientNetB0モデル用

training scripts/
├── janken_train_new_safe.py       # VGG16訓練スクリプト
└── janken_train_improved.py       # EfficientNetB0訓練スクリプト
```

## 結論
VGG16転移学習により、ベースラインから大幅な改善を実現しました（25.4% → 57.63%）。
現在、EfficientNetB0ベースの改良モデルにより、さらなる性能向上を目指しています。

特にパーの認識精度向上が重要な課題であり、データ拡張とファインチューニングによる解決を図っています。
