import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import datetime
import os
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import tensorflow as tf

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)
rng = np.random.default_rng(SEED)

def make_dataset(n=4000, d=10, noise=0.6, imbalance_ratio=0.5):
    X = rng.normal(size=(n, d))
    w = rng.normal(size=(d,))
    b = rng.normal()
    logits = X @ w + b + rng.normal(scale=noise, size=n)
    probs = 1 / (1 + np.exp(-logits))
    
    threshold = np.percentile(probs, (1 - imbalance_ratio) * 100)
    y = (probs > threshold).astype(np.int32)
    
    return X, y.reshape(-1, 1), probs

class EnhancedTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, x_val, y_val, model_name):
        super().__init__()
        self.log_dir = log_dir
        self.x_val = x_val
        self.y_val = y_val
        self.model_name = model_name
        self.writer = tf.summary.create_file_writer(log_dir)
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
            
        with self.writer.as_default():
            for metric_name, metric_value in logs.items():
                tf.summary.scalar(f'metrics/{metric_name}', metric_value, step=epoch)
            
            predictions = self.model.predict(self.x_val, verbose=0)
            pred_classes = (predictions > 0.5).astype(int).flatten()
            true_classes = self.y_val.flatten()
            
            try:
                auc_score = roc_auc_score(true_classes, predictions.flatten())
                tf.summary.scalar('test_metrics/auc', auc_score, step=epoch)
            except:
                pass
            
            if epoch % 5 == 0:
                cm = confusion_matrix(true_classes, pred_classes)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'{self.model_name} - Epoch {epoch}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
                img_buffer.seek(0)
                
                image = tf.image.decode_png(img_buffer.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)
                tf.summary.image("confusion_matrix", image, step=epoch)
                plt.close()
            
            if epoch % 10 == 0:
                try:
                    fpr, tpr, _ = roc_curve(true_classes, predictions.flatten())
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--', label='Random')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'{self.model_name} - ROC Curve (Epoch {epoch})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
                    img_buffer.seek(0)
                    
                    image = tf.image.decode_png(img_buffer.getvalue(), channels=4)
                    image = tf.expand_dims(image, 0)
                    tf.summary.image("roc_curve", image, step=epoch)
                    plt.close()
                except:
                    pass
            
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'kernel'):
                    weights = layer.get_weights()[0]
                    tf.summary.histogram(f'weights/layer_{i}_{layer.name}', weights, step=epoch)
                    tf.summary.scalar(f'weight_stats/layer_{i}_mean', np.mean(weights), step=epoch)
                    tf.summary.scalar(f'weight_stats/layer_{i}_std', np.std(weights), step=epoch)
                    
                    if len(layer.get_weights()) > 1:
                        biases = layer.get_weights()[1]
                        tf.summary.histogram(f'biases/layer_{i}_{layer.name}', biases, step=epoch)
            
            self.writer.flush()

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")
        
        try:
            lr = logs.get("lr", float(self.model.optimizer.learning_rate.numpy()))
        except:
            lr = logs.get("lr", 1e-3)
        
        msg = f"Epoch {epoch + 1:3d}: loss={loss:.4f}"
        if acc is not None:
            msg += f", acc={acc:.4f}"
        if val_loss is not None:
            msg += f", val_loss={val_loss:.4f}"
        if val_acc is not None:
            msg += f", val_acc={val_acc:.4f}"
        msg += f", lr={lr:.2e}"
        print(msg)

class OverfittingEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience=3, gap_loss=0.02, gap_acc=0.02, restore_best_weights=True, verbose=1):
        super().__init__()
        self.patience = int(patience)
        self.gap_loss = float(gap_loss)
        self.gap_acc = float(gap_acc)
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_val = np.inf
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_val = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
            
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")

        if val_loss is not None and val_loss < self.best_val:
            self.best_val = val_loss
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        cond_loss = (
            val_loss is not None and loss is not None and (val_loss - loss) >= self.gap_loss
        )
        cond_acc = (
            acc is not None and val_acc is not None and (acc - val_acc) >= self.gap_acc
        )

        if cond_loss or cond_acc:
            self.wait += 1
            if self.verbose:
                gap_l = (val_loss - loss) if (val_loss is not None and loss is not None) else 0
                gap_a = (acc - val_acc) if (acc is not None and val_acc is not None) else 0
                print(f"  Overfitting detected: loss_gap={gap_l:.4f}, acc_gap={gap_a:.4f} (patience: {self.wait}/{self.patience})")
        else:
            self.wait = 0

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose:
                    print("  Restoring model weights from best val_loss epoch.")
                self.model.set_weights(self.best_weights)
            
            gap_l = (val_loss - loss) if (val_loss is not None and loss is not None) else float("nan")
            gap_a = (acc - val_acc) if (acc is not None and val_acc is not None) else float("nan")
            if self.verbose:
                print(f"  Stopping due to suspected overfitting: loss_gap={gap_l:.4f}, acc_gap={gap_a:.4f}")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Early stopping at epoch {self.stopped_epoch + 1} due to overfitting")

def create_model(input_dim, architecture='deep', regularization=0.01, dropout_rates=None):
    if dropout_rates is None:
        dropout_rates = [0.3, 0.2, 0.1]
    
    if architecture == 'shallow':
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu", 
                             kernel_regularizer=regularizers.l2(regularization)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rates[0]),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
    
    elif architecture == 'medium':
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu", 
                             kernel_regularizer=regularizers.l2(regularization)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rates[0]),
            keras.layers.Dense(32, activation="relu", 
                             kernel_regularizer=regularizers.l2(regularization)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rates[1]),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
    
    elif architecture == 'deep':
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu", 
                             kernel_regularizer=regularizers.l2(regularization)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rates[0]),
            keras.layers.Dense(64, activation="relu", 
                             kernel_regularizer=regularizers.l2(regularization)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rates[1]),
            keras.layers.Dense(32, activation="relu", 
                             kernel_regularizer=regularizers.l2(regularization)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rates[2]),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
    
    return model

def get_optimizers():
    return {
        'Adam': keras.optimizers.Adam(learning_rate=1e-3),
        'AdamW': keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.01),
        'RMSprop': keras.optimizers.RMSprop(learning_rate=1e-3),
        'SGD_Momentum': keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True),
        'Nadam': keras.optimizers.Nadam(learning_rate=1e-3)
    }

def create_callbacks(model_name, optimizer_name, x_val, y_val):
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f"logs/{model_name}_{optimizer_name}_{timestamp}"
    
    callbacks = [
        LossAndErrorPrintingCallback(),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        OverfittingEarlyStopping(patience=3, gap_loss=0.02, gap_acc=0.02, restore_best_weights=True),
        TensorBoard(
            log_dir=log_dir, 
            histogram_freq=5, 
            write_graph=True, 
            write_images=True,
            update_freq='epoch'
        ),
        EnhancedTensorBoardCallback(log_dir, x_val, y_val, f"{model_name}_{optimizer_name}")
    ]
    
    return callbacks, log_dir

def evaluate_model(model, x_test, y_test, model_name="Model"):
    predictions = model.predict(x_test, verbose=0)
    pred_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = y_test.flatten()
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    try:
        auc_score = roc_auc_score(true_classes, predictions.flatten())
    except:
        auc_score = 0.0
    
    cm = confusion_matrix(true_classes, pred_classes)
    
    print(f"\n{model_name} Evaluation:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'auc': auc_score,
        'confusion_matrix': cm,
        'predictions': predictions,
        'pred_classes': pred_classes
    }

def run_experiments():
    print("Creating synthetic dataset...")
    X, y, probs = make_dataset(n=6000, d=15, noise=0.4, imbalance_ratio=0.6)
    
    idx = rng.permutation(len(X))
    train_sz = int(0.7 * len(X))
    val_sz = int(0.15 * len(X))
    
    train_idx = idx[:train_sz]
    val_idx = idx[train_sz:train_sz + val_sz]
    test_idx = idx[train_sz + val_sz:]
    
    x_train, y_train = X[train_idx], y[train_idx]
    x_val, y_val = X[val_idx], y[val_idx]
    x_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Dataset split: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
    print(f"Class distribution - Train: {np.mean(y_train):.3f}, Val: {np.mean(y_val):.3f}, Test: {np.mean(y_test):.3f}")
    
    architectures = ['shallow', 'medium', 'deep']
    optimizers = get_optimizers()
    
    results = {}
    
    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Testing {arch.upper()} architecture")
        print(f"{'='*60}")
        
        arch_results = {}
        
        for opt_name, optimizer in optimizers.items():
            print(f"\nTraining with {opt_name} optimizer...")
            
            model = create_model(x_train.shape[1], architecture=arch)
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            
            model_name = f"{arch}_model"
            callbacks, log_dir = create_callbacks(model_name, opt_name, x_val, y_val)
            
            history = model.fit(
                x_train, y_train,
                batch_size=64,
                epochs=100,
                verbose=0,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
            )
            
            eval_results = evaluate_model(model, x_test, y_test, f"{arch}_{opt_name}")
            eval_results['history'] = history
            eval_results['log_dir'] = log_dir
            
            arch_results[opt_name] = eval_results
        
        results[arch] = arch_results
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    for arch in architectures:
        print(f"\n{arch.upper()} Architecture Results:")
        arch_results = results[arch]
        
        sorted_results = sorted(arch_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for opt_name, result in sorted_results:
            print(f"  {opt_name:12}: Acc={result['accuracy']:.4f}, AUC={result['auc']:.4f}, Loss={result['loss']:.4f}")
    
    best_result = None
    best_score = 0
    best_config = ""
    
    for arch, arch_results in results.items():
        for opt_name, result in arch_results.items():
            score = result['accuracy'] + result['auc']
            if score > best_score:
                best_score = score
                best_result = result
                best_config = f"{arch}_{opt_name}"
    
    print(f"\nBEST OVERALL: {best_config}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"AUC: {best_result['auc']:.4f}")
    print(f"Loss: {best_result['loss']:.4f}")
    
    print(f"\nTensorBoard logs saved in 'logs' directory")
    print("Run 'tensorboard --logdir logs' to view detailed metrics and visualizations")
    
    return results

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    
    print("Starting comprehensive binary classification experiments...")
    print("Testing multiple architectures and optimizers with advanced callbacks")
    
    results = run_experiments()
    
    print("\nExperiment complete! Check TensorBoard for detailed analysis.")
    print("Command: tensorboard --logdir logs")