import os
import numpy as np
import ember

EXE_DIR = "extracted_exe"
X = []
y = []

extractor = ember.PEFeatureExtractor()

for file in os.listdir(EXE_DIR):
    file_path = os.path.join(EXE_DIR, file)

    try:
        with open(file_path, "rb") as f:
            bytez = f.read()

        # Lọc file không phải PE
        if not bytez.startswith(b"MZ"):
            print(f" => Bỏ qua (không phải PE): {file}")
            continue

        feature = extractor.feature_vector(bytez)
        X.append(feature)
        y.append(1)  # Gán nhãn = 1 (malware)
        print(f"[+] Trích xuất đặc trưng: {file}")

    except Exception as e:
        print(f"[!] Lỗi khi xử lý {file}: {e}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.uint8)

np.save("X.npy", X)
np.save("y.npy", y)

print(f"\nĐã lưu {len(X)} đặc trưng hợp lệ vào 'X.npy' và 'y.npy'")
