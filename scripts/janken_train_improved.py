import os
import ssl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

# SSL証明書の問題を回避
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # GPU設定
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU使用可能: {len(gpus) > 0}")
    except Exception as e:
        print(f"GPU設定エラー: {e}")

    # 設定
    target_size = 224
    batch_size = 16
    epochs = 50
    learning_rate = 0.001

    # EfficientNetB0用の前処理関数
    def preprocessing_function(x):
        return tf.keras.applications.efficientnet.preprocess_input(x)

    # より強力なデータ拡張
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0,
        fill_mode='nearest',
        validation_split=0.2
    )

    # 検証用（拡張なし）
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        validation_split=0.2
    )

    # データセット作成
    train_generator = train_datagen.flow_from_directory(
        'img_train',
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        'img_train',
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print(f"訓練データ数: {train_generator.samples}")
    print(f"検証データ数: {validation_generator.samples}")
    print(f"クラス数: {train_generator.num_classes}")
    print(f"クラス名: {train_generator.class_indices}")

    # EfficientNetB0ベースモデル（より軽量で高性能）
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(target_size, target_size, 3)
    )

    # ベースモデルの最後の数層を訓練可能に設定（ファインチューニング）
    for layer in base_model.layers[:-20]:  # 最後の20層以外は凍結
        layer.trainable = False
    for layer in base_model.layers[-20:]:  # 最後の20層は訓練可能
        layer.trainable = True

    # カスタムヘッドを追加
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # より適応的なオプティマイザー設定
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"モデル総パラメータ数: {model.count_params()}")
    trainable_params = sum([tf.size(p).numpy() for p in model.trainable_weights])
    print(f"訓練可能パラメータ数: {trainable_params}")

    # コールバック設定
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'models/janken_model_improved.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # modelsディレクトリが存在しない場合は作成
    os.makedirs('models', exist_ok=True)

    print("訓練開始...")
    try:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )

        # 訓練履歴の保存
        print("訓練履歴を保存中...")
        
        # グラフ作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 精度のグラフ
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy (Improved)', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 損失のグラフ
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss (Improved)', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history_improved.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 最終結果の表示
        print("\n" + "="*60)
        print("訓練完了！")
        print("="*60)
        print(f"最終訓練精度: {history.history['accuracy'][-1]:.4f}")
        print(f"最終検証精度: {history.history['val_accuracy'][-1]:.4f}")
        print(f"最高検証精度: {max(history.history['val_accuracy']):.4f}")
        print("="*60)
        
        # 履歴をnumpyファイルとして保存
        np.save('models/training_history_improved.npy', history.history)
        
        print("モデルと履歴が保存されました:")
        print("- models/janken_model_improved.keras")
        print("- models/training_history_improved.png")
        print("- models/training_history_improved.npy")

    except Exception as e:
        print(f"訓練中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
