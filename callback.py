import tensorflow as tf
from tensorflow import keras
import numpy as np

class OverfittingStoppingCallback(keras.callbacks.Callback):
    
    def __init__(self, max_loss_diff=0.5, patience=3, restore_best_weights=True, verbose=1):
        super().__init__()
        self.max_loss_diff = max_loss_diff
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_val_loss = np.inf
        self.best_epoch = 0
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_val_loss = np.inf
        self.best_epoch = 0
        
        if self.verbose:
            print(f"OverfittingStoppingCallback: מתחיל מעקב אחר overfitting")
            print(f"הפרש מקסימלי מותר: {self.max_loss_diff}")
            print(f"Patience: {self.patience} epochs")
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        if train_loss is None or val_loss is None:
            if self.verbose:
                print(f"Epoch {epoch + 1}: חסרים נתוני loss, לא ניתן לבדוק overfitting")
            return
        
        loss_diff = val_loss - train_loss
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        
        if loss_diff > self.max_loss_diff:
            self.wait += 1
            if self.verbose:
                print(f"Epoch {epoch + 1}: זוהה overfitting! "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"הפרש={loss_diff:.4f} (>{self.max_loss_diff})")
                print(f"Overfitting patience: {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose:
                    print(f"עצירת אימון עקב overfitting מתמשך!")
        else:
            if self.wait > 0 and self.verbose:
                print(f"Epoch {epoch + 1}: שיפור - איפוס overfitting counter. "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"הפרש={loss_diff:.4f}")
            self.wait = 0
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose:
                print(f"האימון נעצר ב-epoch {self.stopped_epoch + 1} עקב overfitting")
                print(f"הטוב ביותר היה ב-epoch {self.best_epoch + 1} עם val_loss={self.best_val_loss:.4f}")
            
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose:
                    print("מחזיר למשקלות הטובים ביותר...")
                self.model.set_weights(self.best_weights)


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    
    model = create_model()
    
    overfitting_callback = OverfittingStoppingCallback(
        max_loss_diff=0.3,
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[overfitting_callback],
        verbose=1
    )
    
    print("האימון הסתיים!")