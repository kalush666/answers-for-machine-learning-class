import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('winequality-red - winequality-red.csv', sep=',')
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={'alcoho': 'alcohol'})

X = df.drop('quality', axis=1)
y = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=150, batch_size=32, validation_split=0.1)
model.evaluate(x_test, y_test)
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)

y_cat = to_categorical(y)
x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

class_model = keras.Sequential()
class_model.add(layers.Dense(256, activation='relu', input_shape=(x_train_c.shape[1],)))
class_model.add(layers.BatchNormalization())
class_model.add(layers.Dropout(0.3))
class_model.add(layers.Dense(128, activation='relu'))
class_model.add(layers.BatchNormalization())
class_model.add(layers.Dropout(0.2))
class_model.add(layers.Dense(64, activation='relu'))
class_model.add(layers.Dense(y_cat.shape[1], activation='softmax'))
class_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
class_model.fit(x_train_c, y_train_c, epochs=150, batch_size=32, validation_split=0.1)
class_model.evaluate(x_test_c, y_test_c)
class_model.summary()

early_stop_cls = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr_cls = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)
class_history = class_model.fit(
	x_train_c, y_train_c,
	validation_data=(x_test_c, y_test_c),
	epochs=150, batch_size=32,
	callbacks=[early_stop_cls, reduce_lr_cls]
)

regression_eval = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Regression MAE: {regression_eval[1]:.4f}")

classification_eval = class_model.evaluate(x_test_c, y_test_c, verbose=0)
print(f"Final Classification Accuracy: {classification_eval[1]:.4f}")
