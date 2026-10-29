
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# False: chỉ dự đoán từ thói quen; True: dùng thêm Height, Weight (gần như tính lại BMI)
DUNG_CHIEU_CAO_CAN_NANG = False

ORDER = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
         'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

# ---------- 1. Đọc dữ liệu, tách X / y, chia train - test ----------
df = pd.read_csv('obesity_level.csv')
print('Kích thước dữ liệu:', df.shape)

X = df.iloc[:, :-1]
y = df.iloc[:, -1].map({n: i for i, n in enumerate(ORDER)}).astype(int)
print('Nhãn:', y.value_counts().sort_index().to_dict())

if not DUNG_CHIEU_CAO_CAN_NANG:
    X = X.drop(columns=['Height', 'Weight'])

X = pd.get_dummies(X, dtype=float)          # mã hoá cột chữ thành cột 0/1
feature_names = list(X.columns)
print('Số đặc trưng sau mã hoá:', len(feature_names))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
with open('config.json', 'w', encoding='utf-8') as f:
    json.dump({'dung_chieu_cao_can_nang': DUNG_CHIEU_CAO_CAN_NANG, 'order': ORDER}, f, ensure_ascii=False)

# ---------- 2. Xây dựng mạng nơ-ron ----------
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(ORDER), activation='softmax'),
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ---------- 3. Huấn luyện và xuất file .h5 ----------
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
          callbacks=[early_stop], verbose=1)

model.save('modelNN.h5')
print('Đã lưu model vào file modelNN.h5')

# ---------- 4. Đánh giá nhanh trên tập test ----------
pred = model.predict(X_test, verbose=0).argmax(axis=1)
print(f'Test Accuracy: {(pred == y_test.values).mean():.4f}')
print(classification_report(y_test, pred, target_names=ORDER, digits=3))
