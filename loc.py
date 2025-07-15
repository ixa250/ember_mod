import os
import pefile

src = "benign_real"
dst = "benign_real_pe"

os.makedirs(dst, exist_ok=True)
count = 0

for fname in os.listdir(src):
    if not fname.lower().endswith(".exe"):
        continue
    try:
        pe = pefile.PE(os.path.join(src, fname))
        count += 1
        os.system(f'copy "{os.path.join(src, fname)}" "{os.path.join(dst, fname)}"')
    except:
        continue

print(f"[+] Đã copy {count} file hợp lệ vào thư mục {dst}")
