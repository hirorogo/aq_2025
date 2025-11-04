import os
import ssl
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import psutil
import logging

# SSL証明書の問題を回避
ssl._create_default_https_context = ssl._create_unverified_context

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設定
TARGET_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 3
CLASS_NAMES = ['0_gu', '1_tyoki', '2_pa']

def check_system_resources():
    """システムリソースを確認"""
    memory = psutil.virtual_memory()
    logger.info(f"メモリ使用率: {memory.percent}% (使用可能: {memory.available // 1024**3} GB)")
    
    if memory.percent > 85:
        logger.warning("メモリ使用率が高いです。他のアプリケーションを終了することを推奨します。")

def setup_gpu():
    """GPU設定を最適化"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU が利用可能です: {len(gpus)} 台")
        else:
            logger.info("GPU が見つかりません。CPUで実行します。")
    except Exception as e:
        logger.warning(f"GPU設定でエラーが発生しました: {e}")

def create_model_architecture():
    """モデルアーキテクチャを作成（学習時と同じ構造）"""
    try:
        # MobileNetV2ベースモデルを作成
        base_model = MobileNetV2(
            input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
            alpha=1.0,
            include_top=False,
            weights='imagenet'
        )
        
        # 上位層を追加
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')
        ])
        
        logger.info("モデルアーキテクチャを作成しました")
        return model
    
    except Exception as e:
        logger.error(f"モデル作成中にエラーが発生しました: {e}")
        raise

def load_trained_model():
    """学習済みモデルを読み込み"""
    model_paths = [
        "models/janken_model_safe.keras",
        "models/janken_model_safe.h5",
        "model.keras",
        "model.h5",
        "janken_model_safe.keras",
        "janken_model_safe.h5"
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
    
    # 既存のモデルがない場合は新しく作成
    logger.warning("既存のモデルが見つかりません。新しいモデルを作成します。")
    model = create_model_architecture()
    
    # モデルをコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_test_data():
    """テストデータを準備"""
    try:
        # テストデータのパスを確認
        test_dir = "img_test"
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"テストディレクトリが見つかりません: {test_dir}")
        
        # ImageDataGeneratorを使用（MobileNetV2の前処理）
        test_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(TARGET_SIZE, TARGET_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        logger.info(f"テストデータを準備しました: {test_generator.samples} サンプル")
        logger.info(f"クラス: {test_generator.class_indices}")
        
        return test_generator
    
    except Exception as e:
        logger.error(f"テストデータ準備中にエラーが発生しました: {e}")
        raise

def evaluate_model(model, test_generator):
    """モデルを評価"""
    try:
        logger.info("モデル評価を開始します...")
        
        # 予測実行
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 真のラベルを取得
        true_classes = test_generator.classes
        
        # 結果をCSVに保存
        np.savetxt("result.csv", predicted_classes, fmt="%d")
        logger.info("予測結果を result.csv に保存しました")
        
        # 評価指標の計算
        accuracy = accuracy_score(true_classes, predicted_classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_classes, predicted_classes, average=None, labels=[0, 1, 2]
        )
        
        # 結果表示
        print("\n" + "="*60)
        print("評価結果")
        print("="*60)
        print(f"\n正解率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nクラスごとの評価指標:")
        print("-" * 60)
        print(f"{'クラス':<15} {'適合率':<12} {'再現率':<12} {'F値':<12} {'サンプル数':<12}")
        print("-" * 60)
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}")
        
        # マクロ平均
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_classes, predicted_classes, average='macro'
        )
        print("-" * 60)
        print(f"{'マクロ平均':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
        
        # 混同行列
        cm = confusion_matrix(true_classes, predicted_classes, labels=[0, 1, 2])
        print("\n混同行列:")
        print("-" * 60)
        print(f"{'':>15}", end="")
        for class_name in CLASS_NAMES:
            print(f"{class_name:>12}", end="")
        print()
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"{class_name:>15}", end="")
            for j in range(len(CLASS_NAMES)):
                print(f"{cm[i][j]:>12}", end="")
            print()
        
        # 詳細レポート
        print("\n詳細な分類レポート:")
        print("-" * 60)
        print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, digits=4))
        
        # 結果をファイルに保存
        save_evaluation_results(accuracy, precision, recall, f1, support, cm, true_classes, predicted_classes)
        
        return accuracy, predicted_classes, true_classes
    
    except Exception as e:
        logger.error(f"モデル評価中にエラーが発生しました: {e}")
        raise

def save_evaluation_results(accuracy, precision, recall, f1, support, cm, true_classes, predicted_classes):
    """評価結果をファイルに保存"""
    try:
        with open("evaluation_result.txt", "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("じゃんけん画像分類 - 評価結果\n")
            f.write("="*60 + "\n\n")
            f.write(f"正解率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            
            f.write("クラスごとの評価指標:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'クラス':<15} {'適合率':<12} {'再現率':<12} {'F値':<12} {'サンプル数':<12}\n")
            f.write("-" * 60 + "\n")
            for i, class_name in enumerate(CLASS_NAMES):
                f.write(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}\n")
            
            # マクロ平均
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_classes, predicted_classes, average='macro'
            )
            f.write("-" * 60 + "\n")
            f.write(f"{'マクロ平均':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}\n\n")
            
            f.write("混同行列:\n")
            f.write("-" * 60 + "\n")
            f.write(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, digits=4))
        
        logger.info("評価結果を evaluation_result.txt に保存しました")
    
    except Exception as e:
        logger.error(f"評価結果保存中にエラーが発生しました: {e}")

def main():
    """メイン関数"""
    try:
        logger.info("じゃんけん画像分類の評価を開始します")
        
        # システムリソース確認
        check_system_resources()
        
        # GPU設定
        setup_gpu()
        
        # モデル読み込み
        model = load_trained_model()
        
        # テストデータ準備
        test_generator = prepare_test_data()
        
        # モデル評価
        accuracy, predicted_classes, true_classes = evaluate_model(model, test_generator)
        
        # 最終結果表示
        print("\n" + "="*60)
        print("評価完了")
        print("="*60)
        print(f"最終精度: {accuracy*100:.2f}%")
        print("結果ファイル:")
        print("- result.csv (予測結果)")
        print("- evaluation_result.txt (詳細な評価結果)")
        print("="*60)
        
        return accuracy
    
    except Exception as e:
        logger.error(f"メイン処理でエラーが発生しました: {e}")
        print(f"\nエラーが発生しました: {e}")
        print("詳細なエラー情報は上記のログを確認してください。")
        return None

if __name__ == "__main__":
    main()
