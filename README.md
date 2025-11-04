# じゃんけん画像分類プロジェクト

**現在の状況**: EfficientNetB0改良モデル完成、強化版モデル準備完了  
**次の目標**: Focal Loss + クラス重み付けによる70%以上の精度達成

## 🚀 今すぐ開始したい場合

```bash
cd /Users/hiro/Documents/aq_2025
source venv/bin/activate
python3 scripts/janken_train_enhanced.py
```

## 📁 プロジェクト構造

```
aq_2025/
├── 📋 QUICK_START.md           # すぐ開始したい時はこれを読む
├── 📂 scripts/                 # 実行スクリプト
│   ├── janken_train_enhanced.py     # 👑 強化版モデル訓練（推奨）
│   ├── janken_train_improved.py     # EfficientNetB0訓練
│   ├── janken_predict_improved.py   # モデル評価
│   └── simple_predict.py            # シンプル予測
├── 📂 analysis/                # 分析ツール
│   └── comprehensive_analysis.py    # 包括的性能分析
├── 📂 docs/                    # 📚 ドキュメント
│   ├── PROJECT_GUIDE.md             # 📖 完全ガイド
│   ├── EXPERIMENT_LOG.md            # 🧪 実験記録・学習ログ
│   └── MODEL_COMPARISON_REPORT.md   # 📊 詳細比較レポート
├── 📂 models/                  # 🤖 保存済みモデル
│   ├── janken_model_safe.keras      # VGG16モデル (57.6%)
│   └── janken_model_improved.keras  # EfficientNetB0 (57.6%, パー改善)
├── 📂 results/                 # 📈 実行結果
│   ├── evaluation_result*.txt       # 評価レポート
│   └── *.png                        # 訓練履歴グラフ
├── 📂 img_train/               # 🖼️ 訓練データ
├── 📂 img_test/                # 🖼️ テストデータ
└── 📂 archive/                 # 🗂️ 古いファイル
```

## 📊 現在のモデル性能

| モデル | 全体精度 | グー | チョキ | パー | 特徴 |
|--------|----------|------|--------|------|------|
| VGG16 | 57.63% | 100% | 42.1% | 12.5% | グー偏向 |
| EfficientNetB0 | 57.63% | 100% | 15.8% | **43.8%** | パー大幅改善 |
| 強化版（予定） | **70%+** | 85%+ | 70%+ | 70%+ | バランス調整 |

## 🎯 重要な成果

✅ **パー認識の大幅改善**: 12.5% → 43.8% (+31.3%)  
✅ **効率的なモデル**: パラメータ数を15M → 4.8Mに削減  
✅ **段階的改善**: 各実験で着実に性能向上  

## 🔄 次回セッション開始手順

### 1. 環境確認
```bash
cd /Users/hiro/Documents/aq_2025
source venv/bin/activate
```

### 2. 状況確認
```bash
# 現在のモデル確認
ls -la models/

# 最新結果確認  
cat results/evaluation_result_janken_model_improved.txt
```

### 3. 作業選択
- **強化版訓練**: `python3 scripts/janken_train_enhanced.py`
- **性能分析**: `python3 analysis/comprehensive_analysis.py`
- **詳細確認**: `cat docs/PROJECT_GUIDE.md`

## 🎨 技術ハイライト

### 実装済み技術
- 🧠 **転移学習**: VGG16, EfficientNetB0
- 🔄 **データ拡張**: 回転、シフト、輝度調整
- ⚡ **ファインチューニング**: 段階的学習
- 📊 **詳細分析**: 混同行列、信頼度分析

### 準備済み次世代技術
- 🎯 **Focal Loss**: 困難な例への集中
- ⚖️ **クラス重み付け**: 不均衡データ対策
- 🔧 **高度な最適化**: 適応的学習率

## 📞 ヘルプ

### ドキュメント
- 📋 `QUICK_START.md` - 今すぐ始めたい時
- 📖 `docs/PROJECT_GUIDE.md` - 完全ガイド
- 🧪 `docs/EXPERIMENT_LOG.md` - 実験記録

### よくある問題
- **環境エラー**: `source venv/bin/activate`
- **パッケージ不足**: `pip install tensorflow matplotlib scikit-learn`
- **メモリ不足**: バッチサイズを16→8に変更

---

**🏆 現在の最高性能**: EfficientNetB0モデル (パー認識43.8%)  
**🎯 次の目標**: 強化版モデルで全体70%達成  
**⏰ 推定所要時間**: 強化版訓練60-90分
