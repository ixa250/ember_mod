import os
import numpy as np
import lief
from ember import PEFeatureExtractor
from tqdm import tqdm

# ======= CẤU HÌNH =======
INPUT_FOLDER = "benign_real_pe"
OUTPUT_FILE = "X_benign_lief.npy"
LOG_FAIL_FILE = "log_lief_fail.txt"

# ======= KHỞI TẠO =======
extractor = PEFeatureExtractor()
features = []
fail_list = []

# ======= HÀM TRÍCH XUẤT =======
def extract_features(path):
    try:
        binary = lief.parse(path)
        if not binary:
            raise ValueError("LIEF parse returned None")

        feats = [
            binary.header.machine,
            binary.header.numberof_sections,
            binary.optional_header.addressof_entrypoint,
            binary.optional_header.imagebase,
            binary.optional_header.subsystem,
            binary.optional_header.dll_characteristics,
            binary.optional_header.sizeof_stack_reserve,
            binary.optional_header.sizeof_heap_reserve,
        ]
        return np.array(feats, dtype=np.float32)

    except Exception:
        try:
            with open(path, "rb") as f:
                bytez = f.read()
            feats = extractor.feature_vector(bytez)
            return feats
        except Exception:
            return None

# ======= TRÍCH XUẤT TOÀN BỘ =======
file_list = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".exe")]
print(f"Đang trích xuất đặc trưng từ {len(file_list)} file...")

for filename in tqdm(file_list):
    file_path = os.path.join(INPUT_FOLDER, filename)
    feats = extract_features(file_path)

    if feats is not None:
        features.append(feats)
    else:
        fail_list.append(filename)

# ======= GHI FILE =======
features = np.array(features, dtype=np.float32)
np.save(OUTPUT_FILE, features)

with open(LOG_FAIL_FILE, "w") as f:
    for name in fail_list:
        f.write(name + "\n")

print(f"\nĐã trích xuất {len(features)} vector → lưu vào {OUTPUT_FILE}")
print(f"{len(fail_list)} file lỗi → ghi vào {LOG_FAIL_FILE}")
