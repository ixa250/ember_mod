import ember
import lightgbm as lgb
import numpy as np
import sys
import os
import subprocess

# Mục đích: Dự đoán xem file .exe có phải là malware hay không bằng cách sử dụng model đã huấn luyện từ ember.
# Lưu ý: Đảm bảo rằng đã cài đặt ember và lightgbm trước khi chạy đoạn code này.

# Màu in terminal
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Đường dẫn tới model đã huấn luyện
MODEL_PATH = "ember_model_trained_2018.txt"

# Folder chứa các file .exe cần test
TEST_FOLDER = "test_samples"

# Tạo file log để lưu kết quả
LOG_FILE = "log.txt"

# Load model
bst = lgb.Booster(model_file=MODEL_PATH)

# Trích xuất đặc trưng
extractor = ember.PEFeatureExtractor()

# Mở file log (ghi đè nếu đã tồn tại)
with open(LOG_FILE, "w") as log:
    log.write("filename,probability,label\n")

# Hàm chặn firewall và xóa file (Windows)
def block_file_in_firewall(file_path):
    try:
        # Ví dụ: Tường lửa có thể chặn đường dẫn cụ thể (chặn ứng dụng khởi chạy ra ngoài mạng)
        subprocess.run(
            ['netsh', 'advfirewall', 'firewall', 'add', 'rule',
             'name=BlockMalware', 'dir=out', 'action=block',
             f'program={file_path}', 'enable=yes'], check=True
        )
        print(f"{RED}[!] Đã thêm rule tường lửa chặn: {file_path}{RESET}")
    except Exception as fw_error:
        print(f"{RED}[!] Không thể thêm rule firewall: {fw_error}{RESET}")

    try:
        os.remove(file_path)
        print(f"{RED}[!] Đã xóa file độc hại: {file_path}{RESET}\n")
    except Exception as del_error:
        print(f"{RED}[!] Không thể xóa file: {del_error}{RESET}\n")

# Duyệt qua từng file .exe trong folder test_samples
for filename in os.listdir(TEST_FOLDER):
    if not filename.endswith(".exe"):
        continue

    file_path = os.path.join(TEST_FOLDER, filename)
    try:
        with open(file_path, "rb") as f:
            bytez = f.read()

        features = np.zeros((1, 2381), dtype=np.float32)
        features[0, :] = extractor.feature_vector(bytez)
        

# Dự đoán
        prediction = float(bst.predict(features)[0])
        label = "MALWARE" if prediction > 0.5 else "BENIGN"
        color = RED if label == "MALWARE" else GREEN
   
# Nếu là malware thì chặn và xóa nó:
        if label == "MALWARE":
            block_file_in_firewall(file_path)            


# In ra màn hình với màu chỉ cho phần số và nhãn
        print(f"[+] Dự đoán cho {filename}:")
        print(f"    Xác suất là malware: {color}{prediction:.4f}{RESET}")
        print(f"    => Phân loại: {color}{label}{RESET}\n")


# Ghi log
        with open(LOG_FILE, "a") as log:
            log.write(f"{filename},{prediction:.4f},{label}\n")
            
            
# Báo lỗi nếu không thể đọc file
    except Exception as e:
        print(f"[!] Lỗi với file {filename}: {str(e)}")