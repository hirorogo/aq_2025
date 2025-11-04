import os
import ssl
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# SSL証明書の問題を回避
ssl._create_default_https_context = ssl._create_unverified_context

# 設定
target_size = 224
batch_size = 8

def preprocessing_function(x):
    """EfficientNetB0用の前処理関数"""
    return tf.keras.applications.efficientnet.preprocess_input(x)

def main():
    # GPU設定
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU設定エラー: {e}")

    # 評価用データセット作成
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    
    test_generator = test_datagen.flow_from_directory(
        "img_test",
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # クラス名を取得
    class_names = list(test_generator.class_indices.keys())
    print(f"クラス名: {class_names}")
    
    # 正解ラベルを取得
    true_labels = test_generator.classes

    # 学習済みモデルロード（優先順位付き）
    model_paths = [
        "models/janken_model_improved.keras",
        "models/janken_model_safe.keras",
        "models/janken_model_safe.h5", 
        "model.keras",
        "model.h5"
    ]
    
    model = None
    model_name = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                model_name = os.path.basename(model_path)
                print(f"{model_path}を読み込みました")
                break
            except Exception as e:
                print(f"{model_path}読み込みエラー: {e}")
                continue
    
    if model is None:
        raise FileNotFoundError("モデルファイルが見つかりません")

    # 予測実施
    test_generator.reset()
    pred_confidence = model.predict(test_generator, verbose=1)
    pred_class = np.argmax(pred_confidence, axis=1)

    # 予測結果ファイル出力
    print("\n予測結果:")
    print(pred_class)
    
    # 信頼度も表示
    print("\n予測信頼度（上位3クラス）:")
    for i, (conf, pred) in enumerate(zip(pred_confidence, pred_class)):
        sorted_indices = np.argsort(conf)[::-1]
        print(f"サンプル{i+1}: 予測={class_names[pred]} ({conf[pred]:.3f}), "
              f"2位={class_names[sorted_indices[1]]} ({conf[sorted_indices[1]]:.3f}), "
              f"3位={class_names[sorted_indices[2]]} ({conf[sorted_indices[2]]:.3f})")
    
    np.savetxt(f"result_{model_name.split('.')[0]}.csv", pred_class, fmt="%d")

    # ========== 評価指標の計算 ==========
    print("\n" + "="*80)
    print(f"評価結果 - モデル: {model_name}")
    print("="*80)
    
    # 正解率
    accuracy = accuracy_score(true_labels, pred_class)
    print(f"\n正解率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 適合率、再現率、F値（マクロ平均とクラスごと）
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_class, average=None, labels=[0, 1, 2]
    )
    
    print("\nクラスごとの評価指標:")
    print("-" * 80)
    print(f"{'クラス':<15} {'適合率':<12} {'再現率':<12} {'F値':<12} {'サンプル数':<12} {'平均信頼度':<12}")
    print("-" * 80)
    
    # 各クラスの平均信頼度を計算
    for i, class_name in enumerate(class_names):
        class_mask = (true_labels == i)
        if np.any(class_mask):
            avg_confidence = np.mean(pred_confidence[class_mask, i])
        else:
            avg_confidence = 0.0
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12} {avg_confidence:<12.4f}")
    
    # マクロ平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_class, average='macro'
    )
    print("-" * 80)
    print(f"{'マクロ平均':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    
    # 重み付き平均
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_class, average='weighted'
    )
    print(f"{'重み付き平均':<15} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f}")
    
    # 混同行列
    cm = confusion_matrix(true_labels, pred_class, labels=[0, 1, 2])
    print("\n混同行列:")
    print("-" * 80)
    print(f"{'実際\\予測':>15}", end="")
    for class_name in class_names:
        print(f"{class_name:>15}", end="")
    print(f"{'再現率':>15}")
    print("-" * 80)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>15}", end="")
        recall_rate = cm[i][i] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0
        print(f"{recall_rate:>15.4f}")
    
    # 適合率の行
    print("-" * 80)
    print(f"{'適合率':>15}", end="")
    for j in range(len(class_names)):
        precision_rate = cm[j][j] / np.sum(cm[:, j]) if np.sum(cm[:, j]) > 0 else 0
        print(f"{precision_rate:>15.4f}", end="")
    print()
    
    # scikit-learnの詳細レポート
    print("\n詳細な分類レポート:")
    print("-" * 80)
    print(classification_report(true_labels, pred_class, target_names=class_names, digits=4))
    
    # 評価結果をファイルに保存
    result_filename = f"evaluation_result_{model_name.split('.')[0]}.txt"
    with open(result_filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"評価結果 - モデル: {model_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"正解率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("クラスごとの評価指標:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'クラス':<15} {'適合率':<12} {'再現率':<12} {'F値':<12} {'サンプル数':<12}\n")
        f.write("-" * 80 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'マクロ平均':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}\n")
        f.write(f"{'重み付き平均':<15} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f}\n\n")
        f.write("詳細な分類レポート:\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(true_labels, pred_class, target_names=class_names, digits=4))
    
    print(f"\n評価結果を {result_filename} に保存しました")
    print("="*80)

if __name__ == "__main__":
    main()
