# 🤖 じゃんけん画像認識AI (AQ 2025)

## 📝 概要
TensorFlow/Kerasを使用したじゃんけん（グー・チョキ・パー）の画像認識AIシステムです。MobileNetV2をベースとした転移学習により、高精度な手形認識を実現しています。

## ✨ 特徴
- **高精度認識**: 89.83%の認識精度を達成
- **MobileNetV2ベース**: 軽量で高速な推論
- **最適化済みデータ拡張**: 遺伝的アルゴリズムで自動調整
- **詳細分析機能**: 失敗ケース分析とレポート生成
- **ブレ対策**: カスタムレイヤーで手ブレに対応

## 🎯 現在の性能指標
- **全体精度**: 89.83%
- **対応クラス**: グー、チョキ、パー
- **最適パラメータ**: 
  - Translation: 10%
  - Brightness: 20%
  - Contrast: 30%
  - Noise: 5%

## 🛠️ 環境構築

### 必要な環境
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

### インストール
```bash
cd /Users/hiro/Documents/aq_2025/janken/janken
pip install -r requirements.txt
```

## 📂 プロジェクト構成

```
aq_2025/
├── janken/janken/
│   ├── janken_train_new.py              # 基本学習スクリプト
│   ├── janken_train_with_subdirs.py     # 高度な学習（推奨）
│   ├── janken_predict_new.py            # 基本予測評価
│   ├── janken_predict_aaa.py            # 詳細レポート生成
│   ├── smart_optimization_search.py     # パラメータ最適化
│   ├── img_train/                       # 学習用画像
│   │   ├── 0_gu/                       # グー画像
│   │   ├── 1_tyoki/                    # チョキ画像
│   │   └── 2_pa/                       # パー画像
│   ├── img_test/                        # テスト用画像
│   └── exhaustive_search_*/             # 最適化結果
└── readme.md
```

## 🚀 クイックスタート

### 1. データ準備
学習用画像を以下の構造で配置：

```
img_train/
├── 0_gu/          # グーの画像ファイル
├── 1_tyoki/       # チョキの画像ファイル
└── 2_pa/          # パーの画像ファイル

img_test/
├── 0_gu/          # テスト用グー画像
├── 1_tyoki/       # テスト用チョキ画像
└── 2_pa/          # テスト用パー画像
```

### 2. モデル学習
```bash
# 高度な学習（データ拡張最適化済み）
python janken_train_with_subdirs.py

# 基本学習
python janken_train_new.py
```

### 3. 予測・評価
```bash
# 詳細レポート生成（推奨）
python janken_predict_aaa.py

# 基本評価
python janken_predict_new.py
```

### 4. パラメータ最適化（オプション）
```bash
# 遺伝的アルゴリズム + 焼きなまし法で最適化
python smart_optimization_search.py
```

## 📊 使用方法詳細

### モデル学習オプション

**`janken_train_with_subdirs.py`** (推奨)
- サブディレクトリ自動検索
- 最適化されたデータ拡張
- カスタムレイヤー対応
- 高度な前処理

**`janken_train_new.py`** (基本)
- シンプルな学習プロセス
- 基本的なデータ拡張
- 軽量実装

### 評価・予測オプション

**`janken_predict_aaa.py`** (推奨)
- 詳細な分析レポート生成
- 失敗ケース可視化
- クラス別精度表示
- 混同行列生成

**`janken_predict_new.py`** (基本)
- 基本的な精度評価
- シンプルな結果表示

## 🔧 カスタマイズ

### データ拡張パラメータの調整
最適化済みパラメータ（`smart_optimization_search.py`の結果）：

```python
OPTIMAL_PARAMS = {
    'rotation': 0.117,
    'zoom': 0.021,
    'translation': 0.094,
    'brightness': 0.447,
    'contrast': 0.428,
    'noise': 0.130
}
```

### モデルアーキテクチャの変更
- `janken_train_with_subdirs.py`内の`create_model()`関数を編集
- MobileNetV2以外のベースモデルに変更可能
- カスタムレイヤーの追加・削除

## 📈 性能改善のヒント

1. **データ品質向上**
   - 多様な背景での撮影
   - 異なる照明条件
   - 手の角度バリエーション

2. **データ拡張最適化**
   - `smart_optimization_search.py`で再最適化
   - 新しいデータセットに応じた調整

3. **アンサンブル学習**
   - 複数モデルの予測結果を統合
   - 精度向上とロバスト性強化

## 🐛 トラブルシューティング

### よくある問題

**Q: メモリエラーが発生する**
A: `BATCH_SIZE`を小さくしてください（例：16 → 8）

**Q: 学習が進まない**
A: 学習率を調整してください（例：0.0001 → 0.001）

**Q: 精度が低い**
A: データの品質確認、データ拡張パラメータの再最適化を行ってください

## 📝 更新履歴

- **2025-10-29**: パラメータ最適化完了（精度89.83%達成）
- **2025-10-XX**: 初期実装完了

## 🤝 貢献

プロジェクトへの貢献を歓迎します：

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 📧 連絡先

質問や提案がありましたら、Issueを作成してください。

---

**AQ 2025 - じゃんけん画像認識AI**  
*最高精度89.83%を誇る実用的なAIシステム*