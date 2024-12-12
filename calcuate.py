import numpy as np
import math

# 入力データ (周囲8箇所のデータ: 海流の角度と速度)
tide_dir = [45, 90, 135, 180, 225, 270, 315, 360]  # 角度 (度)
tide_kt = [2.0, 1.5, 2.5, 3.0, 1.8, 2.2, 2.8, 2.4]  # 速度 (ノット)

# 現在地点からの距離 (例: グリッド上の距離)
distances = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# 時系列データを考慮するための過去の推測値 (初期値をゼロベクトルに設定)
past_prediction = np.array([0.0, 0.0])

# 時間的重み
time_alpha = 0.5  # 現在のデータに対する重み

# ノイズ除去用のガウスフィルタ (簡易)
def gaussian_filter(data, sigma=1.0):
    kernel = np.exp(-np.array(range(-1, 2))**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return np.convolve(data, kernel, mode='same')

# 角度と速度をx, y成分に変換
def polar_to_cartesian(angle_deg, magnitude):
    angle_rad = math.radians(angle_deg)
    return magnitude * math.cos(angle_rad), magnitude * math.sin(angle_rad)

# 加重平均を計算
def weighted_average(vectors, weights):
    vectors = np.array(vectors)
    weights = np.array(weights) / np.sum(weights)  # 重みを正規化
    return np.dot(weights, vectors)

# 周囲8箇所のベクトルを計算
vectors = [polar_to_cartesian(dir, kt) for dir, kt in zip(tide_dir, tide_kt)]

# 距離に基づく重みを計算
weights = [1 / (d**2) for d in distances]

# 現在地点の推測ベクトルを計算
predicted_vector = weighted_average(vectors, weights)

# 時系列データを適用
predicted_vector = time_alpha * np.array(predicted_vector) + (1 - time_alpha) * past_prediction
past_prediction = predicted_vector  # 更新

# 推測結果を極座標に変換
def cartesian_to_polar(x, y):
    magnitude = math.sqrt(x**2 + y**2)
    angle = math.degrees(math.atan2(y, x))
    return angle % 360, magnitude

predicted_angle, predicted_speed = cartesian_to_polar(*predicted_vector)

# 結果を表示
print(f"推測された波の角度: {predicted_angle:.2f}度")
print(f"推測された波の速度: {predicted_speed:.2f}ノット")