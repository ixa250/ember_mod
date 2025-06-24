import ember
import numpy as np
import os
import subprocess
import joblib

# Màu terminal
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Load model mới 
MODEL_PATH = "combined_model.pkl"
TEST_FOLDER = "test_samples"
LOG_FILE = "log_combined.txt"

# Load model huấn luyện (joblib)
model = joblib.load(MODEL_PATH)

# Trích xuất đặc trưng từ PE file
extractor = ember.PEFeatureExtractor()

# Tạo file log
with open(LOG_FILE, "w") as log:
    log.write("filename,probability,label\n")

def block_file_in_firewall(file_path):
    try:
        subprocess.run(
            ['netsh', 'advfirewall', 'firewall', 'add', 'rule',
             'name=BlockMalware', 'dir=out', 'action=block',
             f'program={file_path}', 'enable=yes'], check=True
        )
        print(f"{GREEN}[!] Đã chặn tường lửa: {file_path}{RESET}")
    except Exception as fw_error:
        print(f"{RED}[!] Không thể chặn firewall: {fw_error}{RESET}")

    try:
        os.remove(file_path)
        print(f"{GREEN}[!] Đã xóa file: {file_path}{RESET}\n")
    except Exception as del_error:
        print(f"{RED}[!] Không thể xóa file: {del_error}{RESET}\n")

# Quét toàn bộ file trong test_samples
for filename in os.listdir(TEST_FOLDER):
    if not filename.lower().endswith(".exe"):
        continue

    file_path = os.path.join(TEST_FOLDER, filename)

    try:
        with open(file_path, "rb") as f:
            bytez = f.read()

        # Trích xuất đặc trưng
        features = extractor.feature_vector(bytez).reshape(1, -1)

        # Dự đoán
        prediction = float(model.predict(features)[0])
        label = "MALWARE" if prediction > 0.5 else "BENIGN"
        color = RED if label == "MALWARE" else GREEN

        # In kết quả
        print(f"[+] Dự đoán cho {filename}:")
        print(f"    Xác suất là malware: {color}{prediction:.4f}{RESET}")
        print(f"    => Phân loại: {color}{label}{RESET}\n")

        # Ghi log
        with open(LOG_FILE, "a") as log:
            log.write(f"{filename},{prediction:.4f},{label}\n")

        # Nếu là malware thì block + xóa
        if label == "MALWARE":
            block_file_in_firewall(file_path)

    except Exception as e:
        print(f"[!] Lỗi xử lý file {filename}: {e}")
