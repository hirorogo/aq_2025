import os
import ssl
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# SSL証明書の問題を回避
ssl._create_default_https_context = ssl._create_unverified_context

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設定
TARGET_SIZE = 224
BATCH_SIZE = 8
CLASS_NAMES = ['gu', 'tyoki', 'pa']

def load_model():
    """学習済みモデルを読み込み"""
    model_paths = [
        "models/janken_model_safe.keras",
        "models/janken_model_safe.h5",
        "model.keras",
        "model.h5"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                logger.info(f"モデルを読み込みました: {model_path}")
                return model
            except Exception as e:
                logger.warning(f"モデル読み込みエラー ({model_path}): {e}")
                continue
    
    raise FileNotFoundError("学習済みモデルが見つかりません。先に学習を実行してください。")

def predict_test_images():
    """テスト画像を予測"""
    try:
        # モデル読み込み
        model = load_model()
        
        # テストデータ準備
        test_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )
        
        test_generator = test_datagen.flow_from_directory(
            "img_test",
            target_size=(TARGET_SIZE, TARGET_SIZE),
            batch_size=BATCH_SIZE,
            class_mode=None,  # ラベルなし
            shuffle=False
        )
        
        # 予測実行
        logger.info("予測を実行中...")
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 結果保存
        np.savetxt("result.csv", predicted_classes, fmt="%d")
        
        # 結果表示
        print("\n予測結果:")
        print("-" * 40)
        filenames = test_generator.filenames
        for i, (filename, pred_class, confidence) in enumerate(zip(filenames, predicted_classes, predictions)):
            max_conf = np.max(confidence)
            print(f"{filename:<20} → {CLASS_NAMES[pred_class]:<10} (信頼度: {max_conf:.3f})")
        
        print(f"\n予測完了: {len(predicted_classes)} 枚")
        print("結果を result.csv に保存しました")
        
        return predicted_classes
    
    except Exception as e:
        logger.error(f"予測中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    predict_test_images()
