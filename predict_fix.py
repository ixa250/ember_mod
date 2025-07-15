import ember
import lightgbm as lgb
import numpy as np
import os
import subprocess
import joblib
import sys
import argparse
from send2trash import send2trash  # Thêm thư viện mới

# ========== MÀU ==========
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# ========== THAM SỐ ==========
parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.55, help='Ngưỡng phân loại (default = 0.55)')
parser.add_argument('--debug', action='store_true', help='In vector đặc trưng đầu vào')
args = parser.parse_args()

# ========== CẤU HÌNH ==========
MODEL_PATH = "combined_model.pkl"
TEST_FOLDER = "test_samples"
LOG_FILE = "log_combined.txt"
THRESHOLD = args.threshold

# ========== TẢI MÔ HÌNH ==========
model = joblib.load(MODEL_PATH)
extractor = ember.PEFeatureExtractor()

# ========== KHỞI TẠO LOG ==========
with open(LOG_FILE, "w") as log:
    log.write("filename,probability,label\n")

def try_block_firewall(file_path):
    try:
        subprocess.run(
            ['netsh', 'advfirewall', 'firewall', 'add', 'rule',
             'name=BlockMalware', 'dir=out', 'action=block',
             f'program={file_path}', 'enable=yes'], check=True
        )
        print(f"{GREEN}Đã chặn tường lửa: {file_path}{RESET}")
    except Exception as fw_error:
        print(f"{YELLOW}Không thể chặn firewall (bỏ qua): {fw_error}{RESET}")

# ========== DỰ ĐOÁN ==========
for filename in os.listdir(TEST_FOLDER):
    if not filename.lower().endswith(".exe"):
        continue

    file_path = os.path.join(TEST_FOLDER, filename)

    try:
        with open(file_path, "rb") as f:
            bytez = f.read()

        # Trích xuất đặc trưng
        features = extractor.feature_vector(bytez).reshape(1, -1)

        if args.debug:
            print(f"{YELLOW}[DEBUG] Đặc trưng cho: {filename}{RESET}")
            keys = [
                ("general", 0, 10),
                ("header", 10, 21),
                ("section", 20, 50),
                ("imports", 49, 128),
                ("exports", 177, 128),
                ("strings_metadata", 305, 16),
                ("string_hist", 421, 96),
                ("byte_hist", 549, 256),
                ("2-gram_byte_hist", 805, 256),
                ("sec_hist", 1061, 128),
                ("sec_entropy", 1573, 30),
                ("packer", 2085, 1162)
            ]

            vector = features.flatten()
            for name, start, length in keys:
                values = vector[start:start+length]
                preview = ", ".join([f"{v:.4f}" for v in values[:5]])
                print(f"{GREEN}- {name:<20}:{RESET} [{preview}, ...] (tổng {length})")

        # Dự đoán
        prediction = float(model.predict(features)[0])
        label = "MALWARE" if prediction > THRESHOLD else "BENIGN"
        color = RED if label == "MALWARE" else GREEN

        # In kết quả
        print(f"[+] Dự đoán cho {filename}:")
        print(f"    Xác suất là malware: {color}{prediction:.4f}{RESET}")
        print(f"    => Phân loại: {color}{label}{RESET}\n")

        # Ghi log
        with open(LOG_FILE, "a") as log:
            log.write(f"{filename},{prediction:.4f},{label}\n")

        # Hành động nếu là malware
        if label == "MALWARE":
            try_block_firewall(file_path)
            try:
                send2trash(file_path)
                print(f"{RED}[!] Đã xóa file: {file_path}{RESET}\n")
            except Exception as trash_err:
                print(f"{YELLOW}[!] Không thể xóa file (send2trash): {trash_err}{RESET}\n")

    except Exception as e:
        print(f"{RED}Lỗi xử lý file {filename}: {e}{RESET}")
