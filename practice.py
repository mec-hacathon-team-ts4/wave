import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from joblib import dump  # 【改善点】学習済みモデルを保存するためにjoblibをインポート

# ========== パラメータ設定 ==========
N_PAST = 3    # 過去3ステップ分を特徴に含める
CENTER = (1, 1)  # 中心点のインデックス
neighbors = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),  (0, 0),  (0, 1),
             (1, -1),  (1, 0),  (1, 1)]

# ========== データ読み込み(例) ==========
T = 1000
data = np.random.rand(T,4,4,2) * [360, 2.0]

# ========== 前処理: (dir, kt) -> (u,v) 変換 ==========
dirs = data[:,:,:,0] * np.pi/180.0
kts = data[:,:,:,1]
u_data = kts * np.cos(dirs)
v_data = kts * np.sin(dirs)

# ========== 特徴量とターゲットの生成 ==========
X = []
Y = []  # ターゲットは次時刻(t+1)の中心点(u,v)

for t in range(T - 1 - N_PAST):
    feat = []
    for past_step in range(N_PAST):
        current_t = t + past_step
        for (di, dj) in neighbors:
            i = CENTER[0] + di
            j = CENTER[1] + dj
            feat.append(u_data[current_t, i, j])
            feat.append(v_data[current_t, i, j])
    X.append(feat)

    target_t = t + N_PAST
    Y.append([u_data[target_t+1, CENTER[0], CENTER[1]],
              v_data[target_t+1, CENTER[0], CENTER[1]]])

X = np.array(X)
Y = np.array(Y)

# ========== 学習データとテストデータの分割 ==========
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# ========== モデル学習 ==========
model = Ridge(alpha=1.0)
model.fit(X_train, Y_train)

# ========== 予測 ==========
Y_pred = model.predict(X_test)

# ========== 評価 ==========
mse = mean_squared_error(Y_test, Y_pred)
print("Test MSE:", mse)

# 【改善点】学習済みモデルをjoblibで保存
dump(model, 'trained_model.joblib')
