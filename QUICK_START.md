# クイックスタートガイド

## 🚀 即座に開始する手順

### 1. 環境準備（初回のみ）
```bash
cd /Users/hiro/Documents/aq_2025
source venv/bin/activate
```

### 2. 現在の状況確認
```bash
# モデルファイルの確認
ls -la models/

# 最新の評価結果確認
cat results/evaluation_result_janken_model_improved.txt
```

### 3. 次に実行すべきこと（選択）

#### A. 強化版モデルを訓練（推奨）
```bash
python3 scripts/janken_train_enhanced.py
```
**期待結果**: 70%以上の精度、バランスの取れたクラス認識

#### B. 現在のモデルを評価
```bash
python3 scripts/janken_predict_improved.py
```

#### C. 包括的な分析を実行
```bash
python3 analysis/comprehensive_analysis.py
```

## 📁 整理後のファイル構造

```
aq_2025/
├── scripts/          # メインスクリプト
│   ├── janken_train_enhanced.py      # 強化版訓練（推奨）
│   ├── janken_train_improved.py      # EfficientNetB0訓練
│   ├── janken_predict_improved.py    # 改良モデル評価
│   └── simple_predict.py             # シンプル予測
│
├── analysis/         # 分析スクリプト
│   └── comprehensive_analysis.py     # 包括的分析
│
├── docs/            # ドキュメント
│   ├── PROJECT_GUIDE.md              # 完全ガイド（このファイル）
│   ├── MODEL_COMPARISON_REPORT.md    # モデル比較詳細
│   └── How*.md                       # 旧ガイド
│
├── results/         # 実行結果
│   ├── evaluation_result*.txt        # 評価結果
│   ├── result*.csv                   # CSVレポート
│   └── *.png                         # グラフ
│
├── models/          # 保存モデル
│   ├── janken_model_safe.keras       # VGG16モデル
│   └── janken_model_improved.keras   # EfficientNetB0モデル
│
├── archive/         # 古いファイル
└── img_*/           # データセット
```

## 🎯 現在の優先タスク

### 最高優先（すぐ実行）
1. **強化版モデル訓練**: `python3 scripts/janken_train_enhanced.py`
   - Focal Loss + クラス重み付け
   - 期待精度: 70%以上

### 高優先（今日中）
2. **新モデルの評価**: 訓練完了後に性能測定
3. **結果の比較**: 3つのモデルの性能比較

### 中優先（今週中）
4. **データセット拡張**: より多様な画像の追加
5. **リアルタイム予測**: Webカメラ対応

## ⚠️ 重要な注意点

### 実行前チェック
- 仮想環境がアクティブか: `which python3`
- TensorFlowが利用可能か: `python3 -c "import tensorflow"`
- GPUが認識されているか（オプション）

### 実行時間の目安
- 強化版モデル訓練: 約60-90分
- モデル評価: 約5分
- 包括的分析: 約10分

## 🔄 トラブルシューティング

### よくある問題
1. **ModuleNotFoundError**: `pip install <missing_package>`
2. **Out of Memory**: バッチサイズを16→8に変更
3. **Path Error**: ワーキングディレクトリを確認

### 緊急時のバックアップ
```bash
# 重要ファイルのバックアップ
cp -r models/ models_backup_$(date +%Y%m%d)/
```

---
**最短で結果を得るには**: `python3 scripts/janken_train_enhanced.py` を実行
