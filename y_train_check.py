import numpy as np

# Load y từ EMBER (file .dat)
y_ember = np.memmap("train_features/ember2018/y_train.dat", dtype=np.uint8, mode="r")

# In thông tin
print("Số lượng mẫu trong y_train.dat:", len(y_ember))
print("Phân bố nhãn:", np.bincount(y_ember))
