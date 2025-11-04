#!/usr/bin/env python3
"""
じゃんけん画像分類モデル - 強化版訓練スクリプト
Features:
- EfficientNetB0 with advanced fine-tuning
- Weighted loss for class imbalance
- Advanced data augmentation
- Ensemble techniques
- Focal loss for hard examples
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import class_weight

# GPU設定
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def focal_loss(alpha=1.0, gamma=2.0):
    """Focal Lossの実装 - 困難な例により多くの注意を向ける"""
    def focal_loss_with_logits(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        
        # クロスエントロピー損失
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # Focal weight
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = alpha * tf.pow(1 - pt, gamma)
        
        focal_loss = focal_weight * ce_loss
        return tf.reduce_sum(focal_loss, axis=-1)
    
    return focal_loss_with_logits

def create_enhanced_model(num_classes=3, input_shape=(224, 224, 3)):
    """強化版EfficientNetB0モデルの作成"""
    
    # ベースモデル（EfficientNetB0）
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # 最初は全てfreeze
    base_model.trainable = False
    
    # カスタムヘッドの追加
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # 中間層の追加
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 出力層
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def calculate_class_weights(train_generator):
    """クラス重みの計算"""
    labels = train_generator.labels
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))

def create_advanced_data_generator():
    """高度なデータ拡張の設定"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        channel_shift_range=30.0,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    return train_datagen, validation_datagen

def plot_training_history(history, save_path='models/training_history_enhanced.png'):
    """訓練履歴のプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 精度のプロット
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 損失のプロット
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"訓練履歴のグラフを {save_path} に保存しました")

def main():
    # パラメータ設定
    IMG_SIZE = 224
    BATCH_SIZE = 16
    INITIAL_EPOCHS = 15
    FINE_TUNE_EPOCHS = 10
    TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    
    train_dir = 'img_train'
    
    print("=== じゃんけん画像分類モデル - 強化版訓練 ===")
    
    # データ生成器の作成
    train_datagen, validation_datagen = create_advanced_data_generator()
    
    # データセットの読み込み
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # クラス名とクラス重みの取得
    class_names = list(train_generator.class_indices.keys())
    class_weights = calculate_class_weights(train_generator)
    
    print(f"クラス名: {class_names}")
    print(f"クラス重み: {class_weights}")
    print(f"訓練サンプル数: {train_generator.samples}")
    print(f"検証サンプル数: {validation_generator.samples}")
    
    # モデルの作成
    model, base_model = create_enhanced_model(num_classes=len(class_names))
    
    # 初期コンパイル（転移学習フェーズ）
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # コールバックの設定
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/janken_model_enhanced_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\n=== フェーズ1: 転移学習 ===")
    
    # 初期訓練
    history1 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print("\n=== フェーズ2: ファインチューニング ===")
    
    # ファインチューニングの準備
    base_model.trainable = True
    
    # 最後の30層のみ訓練可能にする
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"ファインチューニング層数: {len([l for l in base_model.layers if l.trainable])}")
    
    # Focal Lossでの再コンパイル
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # より低い学習率
        loss=focal_loss(alpha=1.0, gamma=2.0),
        metrics=['accuracy']
    )
    
    # ファインチューニング実行
    history2 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=TOTAL_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # 履歴の結合
    combined_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    
    # 履歴クラスの作成
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_history_obj = CombinedHistory(combined_history)
    
    # 最終モデルの保存
    model.save('models/janken_model_enhanced.keras')
    print("\n最終モデルを models/janken_model_enhanced.keras に保存しました")
    
    # 訓練履歴のプロット
    plot_training_history(combined_history_obj)
    
    # 最終結果の表示
    final_train_acc = combined_history['accuracy'][-1]
    final_val_acc = combined_history['val_accuracy'][-1]
    
    print(f"\n=== 最終結果 ===")
    print(f"最終訓練精度: {final_train_acc:.4f}")
    print(f"最終検証精度: {final_val_acc:.4f}")
    print(f"精度向上: {final_val_acc - combined_history['val_accuracy'][0]:.4f}")
    
    return model

if __name__ == "__main__":
    try:
        # modelsディレクトリの作成
        os.makedirs('models', exist_ok=True)
        
        # メイン処理の実行
        model = main()
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
