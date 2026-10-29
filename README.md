# Web dự đoán mức béo phì (Flask, chạy cục bộ - không cần ngrok)

## Cấu trúc
- `train.py`   : đọc `obesity_level.csv`, huấn luyện mạng nơ-ron, xuất `modelNN.h5`, `scaler.pkl`, `feature_names.pkl`, `config.json`
- `app.py`     : web Flask (trang giới thiệu `/` và trang dự đoán `/du-doan`)
- `templates/`, `static/` : giao diện tiếng Việt

## Cách chạy (Python 3.10 – 3.12)
```
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py                # chạy 1 lần để tạo modelNN.h5
python app.py                  # rồi mở http://127.0.0.1:5000
```
Muốn dùng thêm chiều cao và cân nặng: đổi `DUNG_CHIEU_CAO_CAN_NANG = True` trong `train.py`, chạy lại `train.py` (web sẽ tự hiện thêm 2 ô nhập, chiều cao tính bằng cm).
