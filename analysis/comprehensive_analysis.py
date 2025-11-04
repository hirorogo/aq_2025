# じゃんけん分類システム - 総合分析スクリプト
# 複数のモデルの性能を比較し、詳細な分析を行う

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_model_safely(model_path):
    """モデルを安全にロードする"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model, True
    except Exception as e:
        print(f"モデル '{model_path}' の読み込みに失敗: {e}")
        return None, False

def get_preprocessor(model_type):
    """モデルタイプに応じた前処理関数を返す"""
    if model_type == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input
    elif model_type == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input
    elif model_type == "mobilenet":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        return lambda x: x / 255.0  # デフォルトの正規化

def evaluate_model(model, model_name, model_type, target_size=224, batch_size=8):
    """単一モデルの評価"""
    print(f"\n{'='*60}")
    print(f"モデル評価: {model_name}")
    print(f"{'='*60}")
    
    # 前処理関数を取得
    preprocess_func = get_preprocessor(model_type)
    
    # テストデータジェネレータ作成
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    test_generator = test_datagen.flow_from_directory(
        "img_test",
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(test_generator.class_indices.keys())
    true_labels = test_generator.classes
    
    # 予測実行
    test_generator.reset()
    pred_confidence = model.predict(test_generator, verbose=0)
    pred_class = np.argmax(pred_confidence, axis=1)
    
    # 精度計算
    accuracy = accuracy_score(true_labels, pred_class)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_class, average=None, labels=[0, 1, 2]
    )
    
    # 結果を辞書で返す
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'class_names': class_names,
        'confusion_matrix': confusion_matrix(true_labels, pred_class, labels=[0, 1, 2]),
        'pred_confidence': pred_confidence,
        'pred_class': pred_class,
        'true_labels': true_labels
    }
    
    print(f"全体精度: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return results

def create_comparison_plots(all_results):
    """比較プロットを作成"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 全体精度比較
    models = [r['model_name'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    
    axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightgreen', 'orange'][:len(models)])
    axes[0, 0].set_title('モデル全体精度比較', fontsize=14)
    axes[0, 0].set_ylabel('精度')
    axes[0, 0].set_ylim(0, 1)
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. クラス別F値比較
    class_names = all_results[0]['class_names']
    x = np.arange(len(class_names))
    width = 0.8 / len(all_results)
    
    for i, result in enumerate(all_results):
        offset = (i - len(all_results)/2 + 0.5) * width
        axes[0, 1].bar(x + offset, result['f1'], width, 
                      label=result['model_name'], alpha=0.8)
    
    axes[0, 1].set_title('クラス別F値比較', fontsize=14)
    axes[0, 1].set_xlabel('クラス')
    axes[0, 1].set_ylabel('F値')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names)
    axes[0, 1].legend()
    
    # 3. 混同行列（最新モデル）
    best_result = max(all_results, key=lambda x: x['accuracy'])
    sns.heatmap(best_result['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title(f'混同行列 - {best_result["model_name"]}', fontsize=14)
    axes[1, 0].set_xlabel('予測')
    axes[1, 0].set_ylabel('実際')
    
    # 4. 性能改善トレンド
    if len(all_results) > 1:
        improvement = []
        baseline = all_results[0]['accuracy']
        for result in all_results:
            improvement.append((result['accuracy'] - baseline) * 100)
        
        axes[1, 1].plot(models, improvement, marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('ベースラインからの改善率', fontsize=14)
        axes[1, 1].set_ylabel('改善率 (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('models/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # GPU設定
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU設定エラー: {e}")
    
    # 評価対象モデルの定義
    models_to_evaluate = [
        {
            'path': 'models/janken_simple_model.h5',
            'name': 'Simple CNN',
            'type': 'simple',
            'target_size': 150
        },
        {
            'path': 'models/janken_model_safe.keras',
            'name': 'VGG16 Transfer',
            'type': 'vgg16',
            'target_size': 224
        },
        {
            'path': 'models/janken_model_improved.keras',
            'name': 'EfficientNetB0',
            'type': 'efficientnet',
            'target_size': 224
        }
    ]
    
    all_results = []
    
    print("じゃんけん分類システム - 総合性能分析")
    print("="*60)
    
    for model_config in models_to_evaluate:
        if os.path.exists(model_config['path']):
            model, success = load_model_safely(model_config['path'])
            if success:
                result = evaluate_model(
                    model, 
                    model_config['name'], 
                    model_config['type'],
                    model_config['target_size']
                )
                all_results.append(result)
            else:
                print(f"モデル '{model_config['name']}' の評価をスキップ")
        else:
            print(f"モデルファイルが見つかりません: {model_config['path']}")
    
    if len(all_results) > 0:
        print("\n" + "="*80)
        print("総合比較結果")
        print("="*80)
        
        # 結果テーブル作成
        print(f"{'モデル名':<20} {'精度':<10} {'マクロF値':<12} {'重み付きF値':<12}")
        print("-" * 60)
        
        for result in all_results:
            macro_f1 = np.mean(result['f1'])
            weighted_f1 = np.average(result['f1'], weights=result['support'])
            print(f"{result['model_name']:<20} {result['accuracy']:<10.4f} {macro_f1:<12.4f} {weighted_f1:<12.4f}")
        
        # 最高性能モデル
        best_model = max(all_results, key=lambda x: x['accuracy'])
        print(f"\n最高性能モデル: {best_model['model_name']}")
        print(f"精度: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)")
        
        # 可視化作成
        create_comparison_plots(all_results)
        print(f"\n比較プロットを保存しました: models/comprehensive_model_comparison.png")
        
        # 詳細結果をファイルに保存
        with open("comprehensive_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write("じゃんけん分類システム - 総合分析レポート\n")
            f.write("="*60 + "\n\n")
            
            for result in all_results:
                f.write(f"モデル: {result['model_name']} ({result['model_type']})\n")
                f.write("-" * 40 + "\n")
                f.write(f"全体精度: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n\n")
                
                f.write("クラス別性能:\n")
                for i, class_name in enumerate(result['class_names']):
                    f.write(f"  {class_name}: 適合率={result['precision'][i]:.4f}, "
                           f"再現率={result['recall'][i]:.4f}, F値={result['f1'][i]:.4f}\n")
                f.write("\n")
        
        print("詳細分析レポートを保存しました: comprehensive_analysis_report.txt")
    
    else:
        print("評価可能なモデルが見つかりませんでした。")

if __name__ == "__main__":
    main()
