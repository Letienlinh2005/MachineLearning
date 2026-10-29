"""Web dự đoán mức béo phì (Flask) - chạy cục bộ, không cần ngrok.

Chạy:  python app.py   rồi mở http://127.0.0.1:5000
Cần có sẵn modelNN.h5, scaler.pkl, feature_names.pkl (tạo bằng notebook hoặc train.py).
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))


def duong_dan(ten):
    return os.path.join(BASE, ten)


model = load_model(duong_dan('modelNN.h5'), compile=False)
scaler = joblib.load(duong_dan('scaler.pkl'))
feature_names = joblib.load(duong_dan('feature_names.pkl'))
ORDER = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
         'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

config = {'dung_chieu_cao_can_nang': 'Height' in feature_names and 'Weight' in feature_names}
NHI_PHAN = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

# ---------- Mô tả các ô nhập ----------
# kind='num': giá trị là số; kind='str': giá trị là chữ (mô hình học theo các mã này)
CO_BAN = [
    {'name': 'Gender', 'label': 'Giới tính', 'type': 'select', 'kind': 'str', 'default': 'Male',
     'options': [('Male', 'Nam'), ('Female', 'Nữ')]},
    {'name': 'Age', 'label': 'Tuổi (năm)', 'type': 'number', 'default': 25, 'min': 10, 'max': 100, 'step': 1},
]
if config['dung_chieu_cao_can_nang']:
    CO_BAN += [
        {'name': 'Height', 'label': 'Chiều cao (cm)', 'type': 'number', 'default': 170, 'min': 100, 'max': 230, 'step': 1},
        {'name': 'Weight', 'label': 'Cân nặng (kg)', 'type': 'number', 'default': 65, 'min': 20, 'max': 250, 'step': 1},
    ]
CO_BAN += [
    {'name': 'family_history_with_overweight', 'label': 'Gia đình có người bị thừa cân', 'type': 'select',
     'kind': 'str', 'default': 'no', 'options': [('yes', 'Có'), ('no', 'Không')]},
]

AN_UONG = [
    {'name': 'FAVC', 'label': 'Thường xuyên ăn đồ nhiều calo (đồ chiên, nước ngọt...)', 'type': 'select',
     'kind': 'str', 'default': 'no', 'options': [('yes', 'Có'), ('no', 'Không')]},
    {'name': 'FCVC', 'label': 'Mức độ ăn rau trong bữa ăn', 'type': 'select', 'kind': 'num', 'default': 2,
     'options': [(1, 'Không bao giờ'), (2, 'Thỉnh thoảng'), (3, 'Luôn luôn')]},
    {'name': 'NCP', 'label': 'Số bữa chính mỗi ngày', 'type': 'select', 'kind': 'num', 'default': 3,
     'options': [(1, '1 bữa'), (2, '2 bữa'), (3, '3 bữa'), (4, '4 bữa trở lên')]},
    {'name': 'CAEC', 'label': 'Ăn vặt giữa các bữa chính', 'type': 'select', 'kind': 'str', 'default': 'Sometimes',
     'options': [('no', 'Không bao giờ'), ('Sometimes', 'Thỉnh thoảng'),
                 ('Frequently', 'Thường xuyên'), ('Always', 'Luôn luôn')]},
    {'name': 'CH2O', 'label': 'Lượng nước uống mỗi ngày', 'type': 'select', 'kind': 'num', 'default': 2,
     'options': [(1, 'Dưới 1 lít'), (2, 'Từ 1 đến 2 lít'), (3, 'Trên 2 lít')]},
    {'name': 'SCC', 'label': 'Có theo dõi lượng calo ăn vào mỗi ngày', 'type': 'select', 'kind': 'str',
     'default': 'no', 'options': [('no', 'Không'), ('yes', 'Có')]},
    {'name': 'CALC', 'label': 'Uống rượu bia', 'type': 'select', 'kind': 'str', 'default': 'no',
     'options': [('no', 'Không bao giờ'), ('Sometimes', 'Thỉnh thoảng'), ('Frequently', 'Thường xuyên')]},
]

SINH_HOAT = [
    {'name': 'FAF', 'label': 'Tần suất tập thể dục / vận động', 'type': 'select', 'kind': 'num', 'default': 1,
     'options': [(0, 'Không vận động'), (1, '1 đến 2 ngày mỗi tuần'),
                 (2, '3 đến 4 ngày mỗi tuần'), (3, '5 ngày mỗi tuần trở lên')]},
    {'name': 'TUE', 'label': 'Thời gian dùng điện thoại, máy tính, TV mỗi ngày', 'type': 'select', 'kind': 'num',
     'default': 1, 'options': [(0, 'Từ 0 đến 2 giờ'), (1, 'Từ 3 đến 5 giờ'), (2, 'Trên 5 giờ')]},
    {'name': 'MTRANS', 'label': 'Phương tiện di chuyển chính', 'type': 'select', 'kind': 'str',
     'default': 'Public_Transportation',
     'options': [('Public_Transportation', 'Phương tiện công cộng (xe buýt, tàu...)'), ('Walking', 'Đi bộ'),
                 ('Automobile', 'Ô tô'), ('Motorbike', 'Xe máy'), ('Bike', 'Xe đạp')]},
    {'name': 'SMOKE', 'label': 'Hút thuốc lá', 'type': 'select', 'kind': 'str', 'default': 'no',
     'options': [('no', 'Không'), ('yes', 'Có')]},
]

NHOM = [('Thông tin cơ bản', CO_BAN), ('Thói quen ăn uống', AN_UONG), ('Vận động và sinh hoạt', SINH_HOAT)]
TAT_CA = [f for _, nhom in NHOM for f in nhom]


NHAN_VI = {
    'Insufficient_Weight': ('Thiếu cân', 'Cân nặng thấp hơn mức khuyến nghị.'),
    'Normal_Weight': ('Cân nặng bình thường', 'Cân nặng nằm trong mức khoẻ mạnh.'),
    'Overweight_Level_I': ('Thừa cân độ I', 'Hơi vượt mức bình thường.'),
    'Overweight_Level_II': ('Thừa cân độ II', 'Vượt mức bình thường rõ rệt.'),
    'Obesity_Type_I': ('Béo phì độ I', 'Mức béo phì nhẹ.'),
    'Obesity_Type_II': ('Béo phì độ II', 'Mức béo phì trung bình.'),
    'Obesity_Type_III': ('Béo phì độ III', 'Mức béo phì nặng.'),
}


def du_doan(nguoi):
    nguoi = dict(nguoi)
    for c in NHI_PHAN:
        if isinstance(nguoi.get(c), str):
            nguoi[c] = 1 if nguoi[c] == 'yes' else 0
    x = pd.get_dummies(pd.DataFrame([nguoi]), dtype=float)
    x = x.reindex(columns=feature_names, fill_value=0.0)
    proba = model.predict(scaler.transform(x), verbose=0)[0]
    return ORDER[int(proba.argmax())], [float(p) * 100 for p in proba]


def doc_form(form):
    """Đọc và kiểm tra dữ liệu người dùng nhập. Trả về (nguoi, loi)."""
    nguoi, loi = {}, []
    for f in TAT_CA:
        raw = (form.get(f['name']) or '').strip()
        if f['type'] == 'number':
            try:
                v = float(raw.replace(',', '.'))
            except ValueError:
                loi.append(f"Vui lòng nhập số cho mục “{f['label']}”.")
                continue
            if not f['min'] <= v <= f['max']:
                loi.append(f"“{f['label']}” nên nằm trong khoảng {f['min']} đến {f['max']}.")
                continue
            nguoi[f['name']] = v / 100 if f['name'] == 'Height' else v   # cm -> mét
        else:
            hop_le = {str(v): v for v, _ in f['options']}
            if raw not in hop_le:
                loi.append(f"Vui lòng chọn một đáp án cho mục “{f['label']}”.")
                continue
            nguoi[f['name']] = float(raw) if f['kind'] == 'num' else raw
    return nguoi, loi


app = Flask(__name__)


@app.route('/')
def gioi_thieu():
    return render_template('gioi_thieu.html', nhan_vi=[NHAN_VI[k] for k in ORDER])


@app.route('/du-doan', methods=['GET', 'POST'])
def trang_du_doan():
    values = {f['name']: str(f['default']) for f in TAT_CA}
    ket_qua, loi = None, []
    if request.method == 'POST':
        values.update({k: v for k, v in request.form.items() if k in values})
        nguoi, loi = doc_form(request.form)
        if not loi:
            nhan, xs = du_doan(nguoi)
            ten, mo_ta = NHAN_VI[nhan]
            ket_qua = {'ten': ten, 'mo_ta': mo_ta,
                       'chi_tiet': [(NHAN_VI[k][0], round(p, 1), k == nhan) for k, p in zip(ORDER, xs)]}
    return render_template('du_doan.html', nhom=NHOM, values=values, ket_qua=ket_qua, loi=loi)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)