# じゃんけん画像分類AIプロジェクト - 使い方ガイド

## 🎯 プロジェクト概要

このプロジェクトは、**シンギュライティバトルクエスト**という大会用のじゃんけん画像分類AIを作成するためのものです。
与えられた手の画像（グー、チョキ、パー）を自動で判別するAIを作り、その精度を競い合います。

### 目標
- 手の画像からじゃんけんの手（グー・チョキ・パー）を正確に分類する
- 高い精度を達成して大会で良い成績を収める
- Python初心者でも理解できるコードで実装する

## 📁 プロジェクト構成

```
aq_2025/
├── janken_train.py         # AIの学習を行うメインファイル
├── janken_train_new_safe.py # 安全版転移学習ファイル（推奨）
├── janken_train_simple.py  # シンプルなCNN学習ファイル
├── janken_predict.py       # 学習したAIで予測を行うファイル
├── janken_predict_new.py   # 修正された予測ファイル
├── janken_predict_fixed.py # 完全修正版予測ファイル（推奨）
├── simple_predict.py      # シンプルな予測専用ファイル
├── requirements.txt        # 必要なライブラリ一覧
├── .gitignore             # Git管理対象外ファイルの設定
├── How to use .md         # このファイル（使い方ガイド）
├── models/                # 学習済みモデル保存フォルダ
├── img_train/             # 学習用画像フォルダ
│   ├── 0_gu/             # グーの画像
│   ├── 1_tyoki/          # チョキの画像
│   └── 2_pa/             # パーの画像
├── img_test/              # テスト用画像フォルダ
│   ├── 0_gu/             # グーの画像（テスト用）
│   ├── 1_tyoki/          # チョキの画像（テスト用）
│   └── 2_pa/             # パーの画像（テスト用）
└── dataset/               # 大会提供のデータセット
```

## 🚀 初期セットアップ（最初に一度だけ）

### 1. 必要な環境の確認

**必要なもの:**
- Python 3.9以上
- MacBook（このプロジェクトはMacで動作することを前提）
- 十分な空き容量（最低5GB以上推奨）
- インターネット接続（ライブラリのダウンロード用）

### 2. Pythonのバージョン確認

ターミナルを開いて以下のコマンドを実行：
```bash
python3 --version
```
`Python 3.9.0`以上が表示されればOKです。

### 3. プロジェクトフォルダに移動

```bash
cd /Users/hiro/Documents/aq_2025
```

### 4. 仮想環境の作成（推奨）

**仮想環境とは：** プロジェクト専用のPython環境を作成して、他のプロジェクトとライブラリが混ざらないようにする仕組みです。

仮想環境を作成：
```bash
python3 -m venv venv
```

仮想環境を有効化（毎回プロジェクトを使用する前に実行）：
```bash
source venv/bin/activate
```

### 5. 必要なライブラリのインストール

**重要：** 仮想環境を有効化した状態で実行してください。

```bash
pip install --only-binary=all numpy matplotlib pillow scikit-learn tensorflow keras opencv-python seaborn pandas psutil
```

**プリコンパイル済みwheelファイルを使用する理由：**
- Python 3.13では一部のライブラリでビルドエラーが発生する可能性があるため
- プリコンパイル済みのファイルを使用することで、安全にインストールできます

**ライブラリが正常にインストールされたか確認：**
```bash
python -c "import numpy; import matplotlib; from PIL import Image; import sklearn; import tensorflow; import keras; import cv2; import seaborn; import pandas; import psutil; print('✅ 全てのライブラリが正常にインストールされました！')"
```

### 6. フォルダ構造の作成

必要なフォルダが存在しない場合は作成：
```bash
mkdir -p models img_train/0_gu img_train/1_tyoki img_train/2_pa
mkdir -p img_test/0_gu img_test/1_tyoki img_test/2_pa
mkdir -p dataset
```

### 7. SSL証明書の更新（重要）

**転移学習を使用するための重要な手順です：**

```bash
# Python証明書の更新
/Applications/Python\ 3.13/Install\ Certificates.command
```

**証明書更新後の確認：**
```bash
python3 -c "import ssl; print('SSL証明書:', ssl.get_default_verify_paths())"
```

## 🔄 プロジェクトの開始方法（毎回実行）

プロジェクトを使用する際は、毎回以下の手順を実行してください：

### 1. プロジェクトフォルダに移動
```bash
cd /Users/hiro/Documents/aq_2025
```

### 2. 仮想環境を有効化
```bash
source venv/bin/activate
```

### 3. 仮想環境が有効化されているか確認
ターミナルのプロンプトが `(venv)` で始まっていれば正常に有効化されています：
```bash
(venv) hiro@hironoMacBook-Air aq_2025 %
```

### 4. プロジェクトの終了
プロジェクトを終了する際は、仮想環境を無効化してください：
```bash
deactivate
```

## 📸 画像データの準備

### 画像の配置方法

#### 学習用画像（`img_train/`フォルダ）
- `img_train/0_gu/`：グーの画像を配置
- `img_train/1_tyoki/`：チョキの画像を配置  
- `img_train/2_pa/`：パーの画像を配置

#### テスト用画像（`img_test/`フォルダ）
- `img_test/0_gu/`：グーの画像を配置（学習に使わないもの）
- `img_test/1_tyoki/`：チョキの画像を配置（学習に使わないもの）
- `img_test/2_pa/`：パーの画像を配置（学習に使わないもの）

### 画像の形式
- **対応形式：** JPG、PNG、JPEG
- **推奨サイズ：** 224×224ピクセル（自動でリサイズされます）
- **推奨枚数：** 各クラス最低100枚以上

### 画像収集のコツ
1. **多様性を重視：** 色々な角度、明るさ、背景で撮影
2. **手の向き：** 左手・右手両方を含める
3. **背景：** 様々な背景で撮影（無地、パターン、屋外など）
4. **照明条件：** 明るい場所、暗い場所、自然光、人工光など

## 🤖 AIの学習方法

### 1. 学習の実行

**重要：** 学習を実行する前に、必ず仮想環境を有効化してください：
```bash
source venv/bin/activate
```

画像データを準備したら、以下のコマンドでAIの学習を開始：

#### 推奨：安全版転移学習を使用（大会参加必須）
```bash
python3 janken_train_new_safe.py
```

**特徴：**
- MobileNetV2を使用した転移学習（大会必須要件）
- 包括的なエラーハンドリング
- SSL証明書問題の自動解決
- メモリ最適化（バッチサイズ8）
- GPU メモリ増加の設定
- 詳細なログと進捗表示

#### シンプルなCNN（学習目的・比較用）
```bash
python3 janken_train_simple.py
```

**特徴：**
- 基本的なCNNアーキテクチャ
- 転移学習を使用しない
- 学習の基礎を理解するのに適している
- 大会には使用不可（転移学習が必須）

#### 従来版（問題が発生した場合の代替）
```bash
python3 janken_train.py
```

### 2. 学習中に表示される情報

```
🎯 じゃんけん画像分類AI - 安全版転移学習
🖥️  システム情報:
   TensorFlow バージョン: 2.20.0
   📱 CPU使用モード
   💾 総メモリ: 8GB
   💾 使用可能メモリ: 4GB

📁 データフォルダを確認中...
  学習用データ:
    ✅ img_train/0_gu: 150枚
    ✅ img_train/1_tyoki: 145枚
    ✅ img_train/2_pa: 142枚

Epoch 1/30
Train Accuracy: 0.6500
Validation Accuracy: 0.7222
```

- **Epoch：** 学習の進行度（1/30は30回中1回目）
- **Train Accuracy：** 学習データでの正答率
- **Validation Accuracy：** 検証データでの正答率

### 3. 修正されたスクリプトの推奨使用手順

#### ステップ1: 安全版転移学習で学習
```bash
python3 janken_train_new_safe.py
```

**期待される結果:**
- 第1エポックで80%以上の検証精度
- 最終的に90%以上のテスト精度
- modelsフォルダにjanken_model_safe.kerasが保存される

#### ステップ2: 修正版スクリプトで評価
```bash
python3 janken_predict_fixed.py
```

**生成されるファイル:**
- `result.csv`: 競技用予測結果
- `evaluation_result.txt`: 詳細な評価レポート
- `models/training_history_safe.png`: 学習履歴グラフ

#### ステップ3: シンプルな予測（オプション）
```bash
python3 simple_predict.py
```

### 4. パフォーマンス指標の理解

#### 競技での評価基準
- **85%以上**: 優秀（上位入賞レベル）
- **70-84%**: 良好（平均以上）
- **50-69%**: 普通（ベースライン）
- **50%未満**: 要改善

#### クラス別精度の確認
各クラス（gu/tyoki/pa）の精度が均等であることが重要：
- 一つのクラスだけ異常に低い場合、データ不足が原因
- 混同行列で間違いやすいパターンを確認

#### 過学習の検出
- 学習精度 > 検証精度 + 10% の場合は過学習
- 早期終了（EarlyStopping）で自動的に防止

### 4. 学習完了時

学習が完了すると以下のファイルが作成されます：
- `models/janken_model.h5`：学習済みモデル（メイン）
- `models/janken_model.keras`：Keras形式モデル
- `model.keras`：予備保存
- `training_results.png`：学習過程のグラフ

### 5. 学習時間の目安

- **CPU使用時：** 約20分〜1時間（安全版では短縮）
- **GPU使用時：** 約5分〜20分（M1/M2 Macの場合）

## 🔮 予測の実行方法

### 1. 推奨：完全修正版を使用（詳細な評価付き）

```bash
python3 janken_predict_fixed.py
```

**特徴：**
- 包括的なエラーハンドリング
- 詳細な評価指標（精度、適合率、再現率、F値）
- 混同行列の表示
- 複数のモデルパスを自動検索
- 結果を評価ファイル（evaluation_result.txt）に保存

### 2. シンプルな予測のみ

```bash
python3 simple_predict.py
```

**特徴：**
- 予測のみに特化
- 軽量で高速
- 結果をresult.csvに保存

### 3. 従来版（修正済み）

```bash
python3 janken_predict_new.py
```

### 4. 予測結果の見方

**janken_predict_fixed.py の出力例：**
```
============================================================
評価結果
============================================================

正解率 (Accuracy): 0.9200 (92.00%)

クラスごとの評価指標:
------------------------------------------------------------
クラス           適合率        再現率        F値          サンプル数   
------------------------------------------------------------
0_gu            0.9500       0.9048       0.9268       21          
1_tyoki         0.9000       0.9000       0.9000       10          
2_pa            0.9167       1.0000       0.9565       11          
------------------------------------------------------------
マクロ平均        0.9222       0.9349       0.9278       

混同行列:
------------------------------------------------------------
                    0_gu       1_tyoki         2_pa
          0_gu           19            2            0
       1_tyoki            1            9            0
          2_pa            0            0           11

詳細な分類レポート:
------------------------------------------------------------
              precision    recall  f1-score   support

        0_gu     0.9500    0.9048    0.9268        21
     1_tyoki     0.9000    0.9000    0.9000        10
        2_pa     0.9167    1.0000    0.9565        11

    accuracy                         0.9200        42
   macro avg     0.9222    0.9349    0.9278        42
weighted avg     0.9222    0.9200    0.9210        42
```

**生成されるファイル：**
- `result.csv`: 予測結果（0, 1, 2の数値）
- `evaluation_result.txt`: 詳細な評価レポート

## 📊 モデルの性能評価

### 1. 精度の確認

学習完了後、以下のような結果が表示されます：

```
=== 最終結果 ===
学習データ精度: 96.5%
検証データ精度: 92.3%
テストデータ精度: 90.1%

=== クラス別精度 ===
グー: 93.2%
チョキ: 89.5%  
パー: 88.9%
```

### 2. 良い性能の目安

- **全体精度：** 85%以上
- **各クラス精度：** 80%以上
- **学習・検証精度の差：** 5%以内（過学習の回避）

## 🛠 トラブルシューティング（詳細なエラーハンドリング）

### ❌ SSL証明書関連エラー

#### エラー例：
```
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1028)>
```

#### 原因：
- MobileNetV2などの事前学習済みモデルのダウンロード時にSSL証明書の検証が失敗

#### 解決方法：
```bash
# 1. Python証明書の更新
/Applications/Python\ 3.13/Install\ Certificates.command

# 2. certifiライブラリの更新
pip install --upgrade certifi

# 3. 証明書の確認
python3 -c "import ssl; print(ssl.get_default_verify_paths())"
```

#### 追加対策：
コード内で以下を追加（`janken_train_new_safe.py`に実装済み）：
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### ❌ メモリ不足エラー

#### エラー例：
```
ResourceExhaustedError: OOM when allocating tensor
tensorflow.python.framework.errors_impl.ResourceExhaustedError
```

#### 原因：
- バッチサイズが大きすぎる
- モデルが複雑すぎる
- 画像サイズが大きすぎる

#### 解決方法：
1. **バッチサイズの削減**（`janken_train_new_safe.py`で実装済み）：
   ```python
   batch_size = 8  # 32から8に削減
   ```

2. **GPU メモリの制限**：
   ```python
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
   ```

3. **他のアプリケーションを閉じる**

4. **システムメモリの確認**：
   ```python
   import psutil
   memory = psutil.virtual_memory()
   print(f"使用可能メモリ: {memory.available // (1024**3)}GB")
   ```

### ❌ データセット読み込みエラー

#### エラー例：
```
FileNotFoundError: No such file or directory: 'img_train'
```

#### 原因：
- 必要なフォルダが存在しない
- 画像ファイルが入っていない

#### 解決方法（`janken_train_new_safe.py`で実装済み）：
```python
def check_data_folders():
    """データフォルダの存在確認"""
    required_folders = [
        "img_train/0_gu", "img_train/1_tyoki", "img_train/2_pa",
        "img_test/0_gu", "img_test/1_tyoki", "img_test/2_pa"
    ]
    
    for folder in required_folders:
        if not os.path.exists(folder):
            print(f"❌ フォルダが存在しません: {folder}")
            return False
        
        count = len([f for f in os.listdir(folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if count == 0:
            print(f"⚠️ {folder}に画像がありません")
            
    return True
```

### ❌ TensorFlow互換性エラー

#### エラー例：
```
AttributeError: module 'tensorflow.keras.utils' has no attribute 'to_categorical'
ImportError: cannot import name 'Adam' from 'tensorflow.keras.optimizers'
```

#### 原因：
- TensorFlowのバージョン不整合
- Python 3.13との互換性問題

#### 解決方法：
1. **TensorFlowの再インストール**：
   ```bash
   pip uninstall tensorflow
   pip install --only-binary=all tensorflow==2.20.0
   ```

2. **互換性チェック**（`janken_train_new_safe.py`で実装済み）：
   ```python
   def check_system_resources():
       print(f"TensorFlow バージョン: {tf.__version__}")
       
       # GPU確認
       gpus = tf.config.list_physical_devices('GPU')
       if gpus:
           print(f"GPU: {len(gpus)}個利用可能")
       else:
           print("CPU使用モード")
   ```

### ❌ 学習停止・精度が上がらない

#### 症状：
- 精度が30%前後で止まる
- 全て同じクラスを予測する
- 学習が進まない

#### 原因と解決方法：

1. **データ不足**：
   ```bash
   # 各クラスの画像数確認
   echo "グー: $(ls img_train/0_gu/ | wc -l)枚"
   echo "チョキ: $(ls img_train/1_tyoki/ | wc -l)枚"
   echo "パー: $(ls img_train/2_pa/ | wc -l)枚"
   ```
   → 各クラス最低50枚以上推奨

2. **データの質の問題**：
   - ぼやけた画像の除去
   - 適切でないラベルの修正
   - 背景の多様化

3. **学習パラメータの調整**（`janken_train_new_safe.py`で実装済み）：
   ```python
   # 早期停止
   EarlyStopping(
       monitor='val_accuracy',
       patience=10,
       restore_best_weights=True
   )
   
   # 学習率削減
   ReduceLROnPlateau(
       monitor='val_accuracy',
       factor=0.5,
       patience=5
   )
   ```

### ❌ 予測時エラー

#### エラー例：
```
ValueError: Input 0 of layer "sequential" is incompatible with the layer
```

#### 原因：
- 学習時と予測時の画像サイズが異なる
- モデルファイルの破損

#### 解決方法：
1. **画像サイズの統一**：
   ```python
   def preprocess_image(image_path):
       image = Image.open(image_path)
       image = image.resize((224, 224))  # 学習時と同じサイズ
       return image
   ```

2. **モデルファイルの確認**：
   ```bash
   ls -la models/janken_model.h5
   # ファイルサイズが0でないか確認
   ```

### ❌ matplotlib表示エラー

#### エラー例：
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

#### 原因：
- GUI環境が利用できない
- バックエンドの設定問題

#### 解決方法（`janken_train_new_safe.py`で実装済み）：
```python
try:
    plt.show()
except:
    print("⚠️ グラフの表示に失敗しましたが、ファイルは保存されました")
```

### 🔧 エラー回避のベストプラクティス

#### 1. 段階的な実行確認
```python
def run_with_error_handling():
    try:
        step1_result = execute_step1()
        print("✅ ステップ1完了")
        
        step2_result = execute_step2()
        print("✅ ステップ2完了")
        
    except Exception as e:
        print(f"❌ エラー発生: {str(e)}")
        return False
    
    return True
```

#### 2. リソース監視
```python
import psutil

def monitor_resources():
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        print("⚠️ メモリ使用量が高いです")
        return False
    return True
```

#### 3. 安全な保存処理
```python
def safe_save_model(model, path):
    try:
        # メイン保存
        model.save(path)
        print(f"✅ モデル保存成功: {path}")
        
        # 予備保存
        backup_path = path.replace('.h5', '_backup.h5')
        model.save(backup_path)
        print(f"✅ 予備保存完了: {backup_path}")
        
    except Exception as e:
        print(f"❌ 保存エラー: {str(e)}")
```

### よくあるエラーと解決方法

#### 1. `ModuleNotFoundError: No module named 'tensorflow'`

**原因：** TensorFlowがインストールされていない、または仮想環境が有効化されていない
**解決方法：**
```bash
# 仮想環境を有効化
source venv/bin/activate
# ライブラリを再インストール
pip install --only-binary=all tensorflow
```

#### 2. `FileNotFoundError: No such file or directory`

**原因：** 画像フォルダが存在しない、または画像が入っていない
**解決方法：**
- フォルダの存在確認
- 画像ファイルの配置確認
- `janken_train_new_safe.py`使用（自動チェック機能付き）

#### 3. 学習が進まない・精度が上がらない

**原因と解決方法：**
- **データ不足：** 各クラス100枚以上の画像を用意
- **データの質：** ぼやけた画像や不適切な画像を除去
- **データの多様性：** 様々な条件で撮影した画像を追加

#### 4. メモリエラー

**原因：** メモリ不足
**解決方法：**
- `janken_train_new_safe.py`を使用（メモリ効率最適化済み）
- 他のアプリケーションを閉じる
- バッチサイズを小さくする

#### 5. Python 3.13での互換性エラー

**原因：** 一部のライブラリがPython 3.13にまだ完全対応していない
**解決方法：**
- 仮想環境を使用する
- プリコンパイル済みwheelファイルを使用する（上記のインストール方法に従う）
- `janken_train_new_safe.py`を使用（互換性対策済み）

### パフォーマンス向上のコツ

#### 1. データの改善
- **Data Augmentation：** 既存画像の回転・反転・明度調整
- **前処理：** 画像の正規化・リサイズの最適化
- **バランス：** 各クラスの画像数を揃える

#### 2. モデルの改善
- **転移学習：** 事前学習済みモデルの活用
- **ハイパーパラメータ調整：** 学習率・バッチサイズの最適化
- **早期停止：** 過学習の防止

## 📈 学習過程の理解

### 1. エポック（Epoch）とは
- データセット全体を1回学習することを「1エポック」と呼ぶ
- 通常20〜100エポック程度学習する
- エポック数が多すぎると過学習のリスク

### 2. 損失（Loss）とは
- AIの予測と正解の差を数値化したもの
- 小さいほど良い（理想は0に近い値）
- 学習が進むにつれて減少するのが正常

### 3. 精度（Accuracy）とは
- 正解した予測の割合（0〜1または0%〜100%）
- 高いほど良い性能
- 学習が進むにつれて上昇するのが正常

## 🎯 大会に向けた準備

### 1. 最終チェック項目

- [ ] 学習データの質と量は十分か
- [ ] 過学習していないか（学習・検証精度の差が小さいか）
- [ ] テストデータで85%以上の精度が出ているか
- [ ] 予測速度は十分速いか
- [ ] SSL証明書が正しく設定されているか
- [ ] 全てのエラーハンドリングが動作するか

### 2. 提出前の確認

- [ ] `janken_predict.py`が正常に動作するか
- [ ] 必要なファイルが全て揃っているか
- [ ] コードにエラーが無いか
- [ ] 転移学習が正常に動作するか

### 3. 禁止事項の再確認

❌ **やってはいけないこと：**
- 外部API（ChatGPT等の生成AI含む）の使用
- 特定データのみを対象とした加工
- 事前に作成した予測結果の埋め込み

✅ **やって良いこと：**
- 転移学習の使用
- Data Augmentationの適用
- 与えられたデータの加工・分割

## 🆘 困ったときの対処法

### 1. エラーが解決しない場合

1. **エラーメッセージを正確に読む**
2. **上記のトラブルシューティングセクションを確認**
3. **安全版スクリプト（`janken_train_new_safe.py`）を使用**
4. **Google検索で解決方法を調べる**
5. **Python・TensorFlowの公式ドキュメントを確認**

### 2. 性能が上がらない場合

1. **データの質を見直す**
2. **`janken_train_new_safe.py`でシステムリソースを確認**
3. **画像の前処理を改善する**
4. **転移学習を適切に使用しているか確認**
5. **エポック数・学習率を調整する**

### 3. コードが理解できない場合

1. **コメントを丁寧に読む**
2. **エラーハンドリング部分から理解を始める**
3. **分からない関数をGoogleで調べる**
4. **小さな部分から動かして理解する**
5. **TensorFlow/Kerasのチュートリアルを参考にする**

## 🔄 推奨実行フロー

### 初回実行時：
1. SSL証明書の更新
2. 仮想環境の作成・有効化
3. ライブラリのインストール
4. データの準備
5. `janken_train_new_safe.py`で学習実行

### 日常的な実行時：
1. 仮想環境の有効化
2. データの追加・更新
3. 学習の実行
4. 結果の確認・改善

## 📚 参考資料

### 学習に役立つサイト
- [TensorFlow公式チュートリアル](https://www.tensorflow.org/tutorials)
- [Keras公式ドキュメント](https://keras.io/)
- [機械学習入門（日本語）](https://www.tensorflow.org/tutorials/keras/classification?hl=ja)

### 画像分類の基礎知識
- CNN（畳み込みニューラルネットワーク）
- Data Augmentation（データ拡張）
- Transfer Learning（転移学習）
- Cross Validation（交差検証）

### エラーハンドリングの参考
- [TensorFlow エラー対処法](https://www.tensorflow.org/guide/function)
- [Python例外処理](https://docs.python.org/ja/3/tutorial/errors.html)

---

**Good Luck! 🎉**

このガイドを参考に、エラーに負けない堅牢なじゃんけんAIを作成してください！
エラーが発生しても、上記のトラブルシューティングを参考に一つずつ解決していけば必ず動作します。