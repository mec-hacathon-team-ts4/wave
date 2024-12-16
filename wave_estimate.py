import numpy as np
from scipy.ndimage import median_filter
import math

# ---- サンプル用疑似データ作成 ----
tide_dir_data = np.array([
    [288.355, 241.173, 345.107, 327.057],
    [8.127, 293.332, 152.043, 137.204],
    [156.781, 205.980, 292.768, 183.410],
    [337.947, 122.561, 322.806, 50.489]
])

tide_kt_data = np.array([
    [0.362, 0.994, 1.203, 0.189],
    [1.854, 1.671, 0.872, 1.516],
    [1.222, 0.953, 0.659, 0.226],
    [0.164, 0.595, 0.814, 1.924]
])

# 推定する箇所
targets = [(1, 1), (1, 2), (2, 1), (2, 2)]

# 周囲8点の座標相対位置
offsets = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]

def estimate(center, tide_dir_data, tide_kt_data):
    neighbors = [(center[0] + dx, center[1] + dy) for dx, dy in offsets]

    # ---- ステップ2：重み計算 (逆距離重み付け) ----
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    distances = [euclidean_distance(center, n) for n in neighbors]
    weights = [1.0 / d for d in distances]

    # ---- ステップ3：角度の加重平均 ----
    neighbor_dirs = [tide_dir_data[n[0], n[1]] for n in neighbors]
    neighbor_rads = [math.radians(d) for d in neighbor_dirs]

    x_sum = sum(w * math.cos(rad) for rad, w in zip(neighbor_rads, weights))
    y_sum = sum(w * math.sin(rad) for rad, w in zip(neighbor_rads, weights))

    weight_total = sum(weights)
    x_mean = x_sum / weight_total
    y_mean = y_sum / weight_total

    estimated_dir = math.degrees(math.atan2(y_mean, x_mean))
    if estimated_dir < 0:
        estimated_dir += 360.0

    # ---- ステップ4：速度の加重平均 ----
    neighbor_speeds = [tide_kt_data[n[0], n[1]] for n in neighbors]
    weighted_speed_sum = sum(v * w for v, w in zip(neighbor_speeds, weights))
    estimated_speed = weighted_speed_sum / weight_total

    return estimated_dir, estimated_speed

# 推定結果を格納する新しい配列を作成
estimated_dir_data = np.copy(tide_dir_data)
estimated_kt_data = np.copy(tide_kt_data)

for target in targets:
    est_dir, est_speed = estimate(target, tide_dir_data, tide_kt_data)
    estimated_dir_data[target] = est_dir
    estimated_kt_data[target] = est_speed



# 出力
print("推定前の角度データ:")
print(tide_dir_data)
print("\n推定前の速度データ:")
print(tide_kt_data)
print("推定後の角度データ:")
print(estimated_dir_data)
print("\n推定後の速度データ:")
print(estimated_kt_data)
