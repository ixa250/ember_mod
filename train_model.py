import ember
import lightgbm as lgb
import os

# Đường dẫn tới thư mục chứa file đặc trưng
data_dir = "train_features/ember2018"
X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")
X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")

print("[+] Training LightGBM model...")
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 64,
    "max_depth": -1,
    "verbose": -1,
}

model = lgb.train(params, d_train, valid_sets=[d_train, d_test], num_boost_round=500)

# Xuất model
model_path = "ember_model_trained.txt"
model.save_model(model_path)
print(f"[+] Model saved to: {model_path}")
