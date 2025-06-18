import ember
import lightgbm as lgb
import numpy as np
import sys

# Đường dẫn tới model đã huấn luyện
MODEL_PATH = "ember_model_trained.txt"

# File cần phân loại
INPUT_FILE = "install.exe"  # Đổi thành file .exe cậu muốn test

# Load model
bst = lgb.Booster(model_file=MODEL_PATH)

# Trích xuất đặc trưng
extractor = ember.PEFeatureExtractor()
with open(INPUT_FILE, "rb") as f:
    bytez = f.read()

features = np.zeros((1, 2381), dtype=np.float32)  # 2381 là số đặc trưng mặc định của EMBER 2018
features[0, :] = extractor.feature_vector(bytez)


# Dự đoán
prediction = bst.predict(features)
print(f"[+] Dự đoán cho {INPUT_FILE}:")
print(f"    Xác suất là malware: {prediction[0]:.4f}")


label = "MALWARE" if prediction[0] > 0.5 else "BENIGN"
print(f"    => Phân loại: {label}")
# Lưu ý: Đảm bảo rằng đã cài đặt ember và lightgbm trước khi chạy đoạn code này.