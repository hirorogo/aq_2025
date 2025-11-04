import os
import ssl
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import psutil
import logging
from datetime import datetime

# SSL証明書の問題を回避
ssl._create_default_https_context = ssl._create_unverified_context

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設定
TARGET_SIZE = 224
BATCH_SIZE = 8  # メモリ不足を防ぐため小さめに設定
NUM_CLASSES = 3
EPOCHS = 50
LEARNING_RATE = 0.0001

def print_banner():
    """開始バナーを表示"""
    print("🎯 じゃんけん画像分類AI - 安全版転移学習")
    print("=" * 60)
    print(f"📅 開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 TensorFlow バージョン: {tf.__version__}")
    
def check_system_resources():
    """システムリソースを確認"""
    memory = psutil.virtual_memory()
    print(f"🖥️  システム情報:")
    print(f"   💾 総メモリ: {memory.total // 1024**3} GB")
    print(f"   💾 使用可能メモリ: {memory.available // 1024**3} GB")
    print(f"   💾 メモリ使用率: {memory.percent}%")
    
    if memory.percent > 85:
        logger.warning("⚠️  メモリ使用率が高いです。他のアプリケーションを終了することを推奨します。")
    
    # GPU確認
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"   🚀 GPU: {len(gpus)} 台利用可能")
            for i, gpu in enumerate(gpus):
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"      GPU {i}: {gpu.name}")
        else:
            print("   📱 CPU使用モード")
    except Exception as e:
        print(f"   ⚠️  GPU設定エラー: {e}")
        print("   📱 CPU使用モード")

def check_data_folders():
    """データフォルダの存在確認"""
    print("\n📁 データフォルダを確認中...")
    
    train_dir = "img_train"
    test_dir = "img_test"
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ 学習用データフォルダが見つかりません: {train_dir}")
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"❌ テスト用データフォルダが見つかりません: {test_dir}")
    
    # 各クラスフォルダの確認
    classes = ['0_gu', '1_tyoki', '2_pa']
    print("  学習用データ:")
    for class_name in classes:
        class_path = os.path.join(train_dir, class_name)
        if os.path.exists(class_path):
            file_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"    ✅ {class_path}: {file_count}枚")
            if file_count < 50:
                logger.warning(f"⚠️  {class_name}の画像数が少ないです（{file_count}枚）。最低100枚推奨。")
        else:
            raise FileNotFoundError(f"❌ クラスフォルダが見つかりません: {class_path}")
    
    print("  テスト用データ:")
    for class_name in classes:
        class_path = os.path.join(test_dir, class_name)
        if os.path.exists(class_path):
            file_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"    ✅ {class_path}: {file_count}枚")
        else:
            logger.warning(f"⚠️  テストクラスフォルダが見つかりません: {class_path}")

def create_data_generators():
    """データジェネレータを作成"""
    print("\n🔄 データ前処理の設定中...")
    
    try:
        # MobileNetV2の前処理関数を使用
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2
        )
        
        test_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )
        
        # 学習用ジェネレータ
        train_generator = train_datagen.flow_from_directory(
            'img_train',
            target_size=(TARGET_SIZE, TARGET_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        # 検証用ジェネレータ
        validation_generator = train_datagen.flow_from_directory(
            'img_train',
            target_size=(TARGET_SIZE, TARGET_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        
        # テスト用ジェネレータ
        test_generator = test_datagen.flow_from_directory(
            'img_test',
            target_size=(TARGET_SIZE, TARGET_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ 学習データ: {train_generator.samples} サンプル")
        print(f"✅ 検証データ: {validation_generator.samples} サンプル")
        print(f"✅ テストデータ: {test_generator.samples} サンプル")
        print(f"✅ クラス: {list(train_generator.class_indices.keys())}")
        
        return train_generator, validation_generator, test_generator
    
    except Exception as e:
        logger.error(f"❌ データジェネレータ作成エラー: {e}")
        raise

def create_model():
    """MobileNetV2ベースの転移学習モデルを作成"""
    print("\n🏗️  転移学習モデルを構築中...")
    
    try:
        # MobileNetV2ベースモデル（ImageNetで事前学習済み）
        base_model = MobileNetV2(
            input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
            alpha=1.0,
            include_top=False,
            weights='imagenet'
        )
        
        # ベースモデルを凍結
        base_model.trainable = False
        
        # 上位層を追加
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(NUM_CLASSES, activation='softmax', name='predictions')
        ])
        
        # モデルをコンパイル
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ モデル構築完了")
        print(f"   📊 総パラメータ数: {model.count_params():,}")
        print(f"   🔒 学習可能パラメータ数: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        return model
    
    except Exception as e:
        logger.error(f"❌ モデル作成エラー: {e}")
        raise

def setup_callbacks():
    """学習用コールバックを設定"""
    print("\n⚙️  学習設定を構成中...")
    
    # modelsディレクトリを作成
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        # 最良モデルを保存
        ModelCheckpoint(
            'models/janken_model_safe.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # 早期終了（過学習防止）
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # 学習率減衰
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("✅ コールバック設定完了")
    return callbacks

def train_model(model, train_generator, validation_generator, callbacks):
    """モデルを学習"""
    print(f"\n🚀 学習開始（最大{EPOCHS}エポック）...")
    print("=" * 60)
    
    try:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ 学習完了！")
        return history
    
    except Exception as e:
        logger.error(f"❌ 学習中にエラーが発生しました: {e}")
        raise

def evaluate_model(model, test_generator):
    """モデルを評価"""
    print("\n📊 モデル評価中...")
    
    try:
        test_generator.reset()
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
        
        print(f"\n🎯 最終結果:")
        print(f"   テスト精度: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   テスト損失: {test_loss:.4f}")
        
        if test_accuracy >= 0.85:
            print("🎉 優秀な精度です！（85%以上）")
        elif test_accuracy >= 0.70:
            print("👍 良い精度です！（70%以上）")
        else:
            print("📈 さらなる改善の余地があります（70%未満）")
        
        return test_accuracy, test_loss
    
    except Exception as e:
        logger.error(f"❌ 評価中にエラーが発生しました: {e}")
        raise

def save_training_history(history):
    """学習履歴をグラフとして保存"""
    print("\n📈 学習履歴を保存中...")
    
    try:
        # 精度のグラフ
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # 損失のグラフ
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history_safe.png', dpi=300, bbox_inches='tight')
        print("✅ 学習履歴グラフを保存しました: models/training_history_safe.png")
        
    except Exception as e:
        logger.warning(f"⚠️  グラフ保存でエラーが発生しました: {e}")

def main():
    """メイン処理"""
    try:
        # バナー表示
        print_banner()
        
        # システムリソース確認
        check_system_resources()
        
        # データフォルダ確認
        check_data_folders()
        
        # データジェネレータ作成
        train_gen, val_gen, test_gen = create_data_generators()
        
        # モデル作成
        model = create_model()
        
        # コールバック設定
        callbacks = setup_callbacks()
        
        # 学習実行
        history = train_model(model, train_gen, val_gen, callbacks)
        
        # 最良モデルを読み込み
        try:
            best_model = tf.keras.models.load_model('models/janken_model_safe.keras')
            print("✅ 最良モデルを読み込みました")
        except:
            best_model = model
            print("⚠️  最良モデルの読み込みに失敗。現在のモデルを使用します。")
        
        # 評価実行
        test_accuracy, test_loss = evaluate_model(best_model, test_gen)
        
        # 学習履歴保存
        save_training_history(history)
        
        # 最終結果表示
        print("\n" + "=" * 60)
        print("🎊 学習プロセス完了！")
        print("=" * 60)
        print(f"📅 完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 最終テスト精度: {test_accuracy*100:.2f}%")
        print(f"💾 保存されたファイル:")
        print(f"   • models/janken_model_safe.keras (最良モデル)")
        print(f"   • models/training_history_safe.png (学習履歴グラフ)")
        print("\n🔮 予測を実行するには:")
        print("   python3 janken_predict_fixed.py")
        print("=" * 60)
        
        return test_accuracy
    
    except Exception as e:
        logger.error(f"❌ メイン処理でエラーが発生しました: {e}")
        print(f"\n💥 エラーが発生しました: {e}")
        print("\n🔧 トラブルシューティング:")
        print("1. 仮想環境がアクティブかを確認")
        print("2. 必要なライブラリがインストールされているかを確認")
        print("3. img_train/フォルダに画像があるかを確認")
        print("4. メモリ不足の場合は他のアプリケーションを終了")
        return None

if __name__ == "__main__":
    main()
