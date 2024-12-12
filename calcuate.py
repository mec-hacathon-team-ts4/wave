import numpy as np
from scipy.ndimage import median_filter
import math

# ---- サンプル用疑似データ作成 ----
# 4x4の海流角度データ(0〜359度)と速度データ(0〜2ノット程度)をランダム生成
np.random.seed(0)  # 再現性のため

# 入力データ (周囲8箇所のデータ: 海流の角度と速度)
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

# 中心点(1,1)を推定したい
center = (1, 1)

# 周囲8点の座標
neighbors = [
    (0,0), (0,1), (0,2),
    (1,0),         (1,2),
    (2,0), (2,1), (2,2)
]

# ---- ステップ2：重み計算 (逆距離重み付け) ----
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

distances = [euclidean_distance(center, n) for n in neighbors]
weights = [1.0/d for d in distances]

# ---- ステップ3：角度の加重平均 ----
# 周囲角度取得
neighbor_dirs = [tide_dir_data[n[0], n[1]] for n in neighbors]

# 度→ラジアン変換
neighbor_rads = [math.radians(d) for d in neighbor_dirs]

# 単位ベクトル化＆重み付け
x_sum = 0.0
y_sum = 0.0
for rad, w in zip(neighbor_rads, weights):
    x_sum += w * math.cos(rad)
    y_sum += w * math.sin(rad)

# 合計ベクトルから平均ベクトルを求める
weight_total = sum(weights)
x_mean = x_sum / weight_total
y_mean = y_sum / weight_total

# atan2で角度（ラジアン）を求め、度に戻す
estimated_dir = math.degrees(math.atan2(y_mean, x_mean))
if estimated_dir < 0:
    estimated_dir += 360.0

# ---- ステップ4：速度の加重平均 ----
neighbor_speeds = [tide_kt_data[n[0], n[1]] for n in neighbors]

weighted_speed_sum = sum(v*w for v, w in zip(neighbor_speeds, weights))
estimated_speed = weighted_speed_sum / weight_total

print("推定角度（フィルタ前）:", estimated_dir, "度")
print("推定速度（フィルタ前）:", estimated_speed, "ノット")

# ---- ステップ5：ノイズ除去（オプション）----
# メディアンフィルタをかけてから再計算(あくまで例)
filtered_dir_data = median_filter(tide_dir_data, size=3)
filtered_kt_data  = median_filter(tide_kt_data,  size=3)

neighbor_dirs_f = [filtered_dir_data[n[0], n[1]] for n in neighbors]
neighbor_rads_f = [math.radians(d) for d in neighbor_dirs_f]

x_sum_f = 0.0
y_sum_f = 0.0
for rad, w in zip(neighbor_rads_f, weights):
    x_sum_f += w * math.cos(rad)
    y_sum_f += w * math.sin(rad)

x_mean_f = x_sum_f / weight_total
y_mean_f = y_sum_f / weight_total
estimated_dir_f = math.degrees(math.atan2(y_mean_f, x_mean_f))
if estimated_dir_f < 0:
    estimated_dir_f += 360.0

neighbor_speeds_f = [filtered_kt_data[n[0], n[1]] for n in neighbors]
weighted_speed_sum_f = sum(v*w for v, w in zip(neighbor_speeds_f, weights))
estimated_speed_f = weighted_speed_sum_f / weight_total

print("推定角度（フィルタ後）:", estimated_dir_f, "度")
print("推定速度（フィルタ後）:", estimated_speed_f, "ノット")

# ---- ステップ6：差分解析による異常検知（例）----
# フィルタ後の推定値と周囲(フィルタ後)の平均差分を見てみる
diff_dirs = []
for d in neighbor_dirs_f:
    # 角度差は360度周期性に注意する必要があるが、ここでは簡易に絶対差分最小値を利用
    diff_raw = abs(estimated_dir_f - d)
    diff_angle = min(diff_raw, 360 - diff_raw)  # 角度差は小さい方を取る
    diff_dirs.append(diff_angle)

mean_dir_diff = np.mean(diff_dirs)
if mean_dir_diff > 45:  # 45度以上の平均差分なら大きなずれとして警告(例)
    print("警告: 周囲との角度差が大きいです。異常の可能性あり。")

speed_diffs = [abs(estimated_speed_f - s) for s in neighbor_speeds_f]
mean_speed_diff = np.mean(speed_diffs)
if mean_speed_diff > 1.0:  # 1ノット以上乖離していたら警告
    print("警告: 周囲との速度差が大きいです。異常の可能性あり。")