import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

tf.config.run_functions_eagerly(True)

def load_and_preprocess_data():
    df = pd.read_csv('winequality-red - winequality-red.csv', sep=',')
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={'alcoho': 'alcohol'})
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['quality'].value_counts().sort_index()}")
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def create_optimizers():
    return {
        'Adam': keras.optimizers.Adam(learning_rate=0.001),
        'SGD': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        'AdamW': keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    }

def create_regression_model(input_shape):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,),
                    kernel_regularizer=regularizers.L2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.L2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    return model

def create_classification_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,),
                    kernel_regularizer=regularizers.L2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.L2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
    ]

def train_and_evaluate_models():
    X_scaled, y, scaler = load_and_preprocess_data()
    
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    y_cat = to_categorical(y)
    x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42
    )
    
    optimizers = create_optimizers()
    regression_results = {}
    classification_results = {}
    
    print("=" * 50)
    print("TRAINING REGRESSION MODELS")
    print("=" * 50)
    
    for opt_name, optimizer in optimizers.items():
        print(f"\n🔄 Training Regression with {opt_name}...")
        
        model = create_regression_model(x_train.shape[1])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        callbacks = create_callbacks()
        
        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
        
        regression_results[opt_name] = {
            'loss': test_loss,
            'mae': test_mae,
        }
        
        print(f"✅ {opt_name}: MAE={test_mae:.4f}, Loss={test_loss:.4f}")
    
    print("\n" + "=" * 50)
    print("TRAINING CLASSIFICATION MODELS")
    print("=" * 50)
    
    for opt_name, optimizer in optimizers.items():
        print(f"\n🔄 Training Classification with {opt_name}...")
        
        model = create_classification_model(x_train_c.shape[1], y_cat.shape[1])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = create_callbacks()
        
        history = model.fit(
            x_train_c, y_train_c,
            validation_data=(x_test_c, y_test_c),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        test_loss, test_acc = model.evaluate(x_test_c, y_test_c, verbose=0)
        
        classification_results[opt_name] = {
            'loss': test_loss,
            'accuracy': test_acc,
        }
        
        print(f"✅ {opt_name}: Accuracy={test_acc:.4f}, Loss={test_loss:.4f}")
    
    return regression_results, classification_results

def print_final_results(regression_results, classification_results):
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    best_reg = min(regression_results.items(), key=lambda x: x[1]['mae'])
    best_cls = max(classification_results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\n📈 BEST REGRESSION MODEL: {best_reg[0]}")
    print(f"   MAE: {best_reg[1]['mae']:.4f}")
    print(f"   Loss: {best_reg[1]['loss']:.4f}")
    
    print(f"\n📊 BEST CLASSIFICATION MODEL: {best_cls[0]}")
    print(f"   Accuracy: {best_cls[1]['accuracy']:.4f}")
    print(f"   Loss: {best_cls[1]['loss']:.4f}")
    
    print(f"\n📋 ALL REGRESSION RESULTS:")
    print("-" * 40)
    for opt_name, results in sorted(regression_results.items(), key=lambda x: x[1]['mae']):
        print(f"   {opt_name:10}: MAE={results['mae']:.4f}")
    
    print(f"\n📋 ALL CLASSIFICATION RESULTS:")
    print("-" * 40)
    for opt_name, results in sorted(classification_results.items(), 
                                   key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"   {opt_name:10}: Acc={results['accuracy']:.4f}")

if __name__ == "__main__":
    print("🍷 Wine Quality Prediction - Fast Version")
    print("==========================================")
    
    regression_results, classification_results = train_and_evaluate_models()
    print_final_results(regression_results, classification_results)
    
    print(f"\n✅ Training completed!")
