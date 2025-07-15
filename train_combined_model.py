import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split

# Màu in terminal
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Load dataset đã gộp
X = np.load("X_combined.npy")
y = np.load("Y_combined.npy")

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train mô hình LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'verbose': -1
}

print("Training LightGBM model...")
model = lgb.train(params, train_data, valid_sets=[train_data, test_data], num_boost_round=500)

# Lưu model
joblib.dump(model, "combined_model.pkl")
print("Huấn luyện hoàn tất. Model đã lưu vào: combined_model.pkl")
