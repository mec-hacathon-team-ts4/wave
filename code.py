import time
import math
import numpy as np
from joblib import load

# ========== ユーザーパラメータ ==========
N_PAST = 1  # 過去何ステップ分を使用するか（例：1なら現在のみ、3なら過去2回分+現在）
CENTER = (1, 1)
neighbors = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),  (0, 0),  (0, 1),
             (1, -1),  (1, 0),  (1, 1)]

# ========== 学習済みモデル読み込み ==========
model = load('trained_model.joblib')  # 前回学習済みモデルを保存したファイル

# 前処理用の平均・標準偏差がある場合はここで読み込む
# mean = np.load('mean.npy')
# std = np.load('std.npy')

# 過去データ保持用 (N_PAST>1の場合に備えた例)
past_features = []

def get_current_wave_data():
    """
    現在の4x4海洋データを取得する関数(ダミー実装)。
    実際にはセンサーAPIやデータベースからの取得処理を書く。
    tide_dir: 角度 [0,360)
    tide_kt:  速度(knot)
    """
    tide_dir = np.random.rand(4,4)*360.0
    tide_kt = np.random.rand(4,4)*2.0
    return tide_dir, tide_kt

def compute_uv(tide_dir, tide_kt):
    # (dir, kt)->(u,v)変換
    dir_rad = np.deg2rad(tide_dir)
    u = tide_kt * np.cos(dir_rad)
    v = tide_kt * np.sin(dir_rad)
    return u, v

def build_feature_vector(u_data, v_data, past_features):
    """
    現在のデータ(u,v)から特徴ベクトルXを作成。
    N_PAST>1の場合はpast_featuresで過去データを含める。
    """
    current_feat = []
    for (di, dj) in neighbors:
        i = CENTER[0] + di
        j = CENTER[1] + dj
        current_feat.append(u_data[i,j])
        current_feat.append(v_data[i,j])

    # 過去データと結合
    # N_PAST=1なら過去なしでcurrent_featのみ
    # N_PAST>1なら過去の特徴量をつなげる
    if N_PAST == 1:
        X = np.array(current_feat)
    else:
        # 過去データありの場合
        # 過去データは[(feature_vector at t-1), (t-2), ...]の形で保持する想定
        # 今回は簡易的実装例
        all_feats = []
        # 全ての過去N_PAST-1回分
        for pf in past_features:
            all_feats.extend(pf)
        # 最後に現在の特徴を付け足す
        all_feats.extend(current_feat)
        X = np.array(all_feats)

    # 前処理(必要なら)
    # X = (X - mean) / std

    return X

def main_loop():
    while True:
        # 1. 最新データ取得(4x4のtide_dir, tide_kt)
        tide_dir, tide_kt = get_current_wave_data()

        # 2. (u,v)変換
        u_data, v_data = compute_uv(tide_dir, tide_kt)

        # 3. 特徴量ベクトル作成
        X = build_feature_vector(u_data, v_data, past_features)

        # 4. 予測
        Y_pred = model.predict([X])  # 出力は(u,v)予測、shape=(1,2)
        u_pred, v_pred = Y_pred[0]

        # 5. 結果表示
        pred_speed = math.sqrt(u_pred**2 + v_pred**2)
        pred_dir = math.degrees(math.atan2(v_pred, u_pred))
        if pred_dir < 0:
            pred_dir += 360.0

        print("【予測結果】")
        print(f"1分後のu={u_pred:.3f}, v={v_pred:.3f}")
        print(f"1分後の速度(kt)={pred_speed:.3f}, 方向(deg)={pred_dir:.3f}\n")

        # 過去特徴量更新(N_PAST>1の場合)
        # 今回はN_PAST=1なので特に過去保持しないが、必要なら以下のような処理を行う
        # if N_PAST > 1:
        #     # 現在の特徴量current_featを格納
        #     current_feat = X[-18:] # 最後の現在ステップ分を抽出(9点*2=18特徴)
        #     # キュー操作で過去N-1個保持
        #     past_features.append(current_feat)
        #     if len(past_features) > N_PAST-1:
        #         past_features.pop(0)

        # 6. 60秒待機
        time.sleep(60)


if __name__ == "__main__":
    main_loop()
