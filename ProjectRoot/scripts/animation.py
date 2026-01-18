# %%
import numpy as np
import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import math

# =========================================================
#  設定エリア
# =========================================================
MATRICES_DIR = "../system_matrices"
POINTS_CSV = "../data/2d_points.csv"
BONES_CONFIG = "../data/bones_config.json"

# 座標軸の設定
INVERT_Y_AXIS = True 
ANIMATION_INTERVAL = 50

# 日本語フォント設定（文字化け対策）
try:
    import japanize_matplotlib
except ImportError:
    import platform
    system = platform.system()
    # OSごとの標準的な日本語フォント候補
    fonts = []
    if system == "Windows":
        fonts = ["MS Gothic", "Meiryo", "Yu Gothic"]
    elif system == "Darwin": # Mac
        fonts = ["AppleGothic", "Hiragino Sans"]
    else: # Linux
        fonts = ["Noto Sans CJK JP", "TakaoGothic", "IPAGothic"]
    
    for f in fonts:
        try:
            plt.rcParams['font.family'] = f
            break
        except:
            continue

# =========================================================
#  3D復元クラス
# =========================================================
class MultiViewTriangulator:
    def __init__(self, matrices_dir):
        self.projections = {}
        self.camera_map = {}
        self.load_matrices(matrices_dir)

    def load_matrices(self, matrices_dir):
        if not os.path.exists(matrices_dir):
            print(f"Error: Directory {matrices_dir} not found.")
            sys.exit(1)
        for f in os.listdir(matrices_dir):
            if f.endswith("_projection.npy"):
                cam = f.replace("_projection.npy", "")
                self.projections[cam] = np.load(os.path.join(matrices_dir, f))
                self.camera_map[cam.lower()] = cam

    def get_projection(self, csv_cam_name):
        return self.projections.get(self.camera_map.get(csv_cam_name.lower()))

    def triangulate_n_views(self, cam_names, points_2d):
        if len(cam_names) < 2: return None
        A = []
        for cam, (u, v) in zip(cam_names, points_2d):
            P = self.get_projection(cam)
            if P is None or np.isnan(u) or np.isnan(v): continue
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
        if len(A) < 4: return None
        try:
            u_svd, s_svd, vh_svd = np.linalg.svd(np.array(A))
            X = vh_svd[-1]
            return X[:3] / X[3]
        except: return None



# =========================================================
#  OneEuroFilter クラス (ジッター除去・スムージング用)
# =========================================================
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: 低速時のカットオフ周波数 (小さいほど滑らかだが遅れる)
        beta: 速度係数 (大きいほど高速動作時の追従性が上がる)
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        
        # タイムスタンプが重複した場合の回避
        if t_e <= 0: return self.x_prev

        # 信号の変化率(速度)を推定
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 速度に応じてカットオフ周波数を動的に変更
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        # フィルタリング実行
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

# =========================================================
#  フィルタ適用関数 (ani3.pyの既存ロジックに組み込む)
# =========================================================
def apply_temporal_smoothing(results, frames):
    """
    3D復元結果(results)に対してOneEuroFilterを適用する
    results: {frame_idx: {part_name: [x, y, z]}}
    """
    print("Applying OneEuroFilter smoothing...")
    
    # 部位ごとにフィルタインスタンスを作成
    filters = {}
    
    # 最初のフレームにある部位で初期化
    if not frames: return results
    
    sorted_frames = sorted(frames)
    first_frame = sorted_frames[0]
    
    # パラメータ調整推奨値:
    # min_cutoff: 0.1 ~ 1.0 (小さいほどブレない)
    # beta: 0.001 ~ 0.1 (大きいほど速い動きに遅れない)
    MIN_CUTOFF = 0.5 
    BETA = 0.05       
    
    smoothed_results = {}

    for i, f_num in enumerate(sorted_frames):
        smoothed_results[f_num] = {}
        current_pose = results.get(f_num, {})
        
        # タイムスタンプ（秒単位と仮定。FPS=30なら 1/30 ずつ増える）
        # 正確な時間が不明ならフレーム番号でも動作はするが、パラメータ調整が必要
        t = i * (1.0 / 30.0) 

        for part, pos in current_pose.items():
            val = np.array(pos)
            
            if part not in filters:
                # 初回出現時は初期化
                filters[part] = [
                    OneEuroFilter(t, val[0], min_cutoff=MIN_CUTOFF, beta=BETA), # X
                    OneEuroFilter(t, val[1], min_cutoff=MIN_CUTOFF, beta=BETA), # Y
                    OneEuroFilter(t, val[2], min_cutoff=MIN_CUTOFF, beta=BETA)  # Z
                ]
                smoothed_results[f_num][part] = val
            else:
                # フィルタ適用
                new_x = filters[part][0](t, val[0])
                new_y = filters[part][1](t, val[1])
                new_z = filters[part][2](t, val[2])
                smoothed_results[f_num][part] = np.array([new_x, new_y, new_z])
                
    return smoothed_results, frames        

# =========================================================
#  データロード
# =========================================================
def load_data():
    if not os.path.exists(POINTS_CSV):
        print(f"Error: {POINTS_CSV} not found.")
        sys.exit(1)
    try:
        df = pd.read_csv(POINTS_CSV)
        rename_map = {'SyncFrame': 'frame', 'Camera': 'camera', 'Part': 'body_part', 'X': 'x', 'Y': 'y'}
        df.rename(columns=rename_map, inplace=True)
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['frame'] = pd.to_numeric(df['frame'], errors='coerce').fillna(-1).astype(int)
        df = df[df['frame'] >= 0].dropna(subset=['x', 'y'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    connections = []
    if os.path.exists(BONES_CONFIG):
        with open(BONES_CONFIG, 'r', encoding='utf-8') as f:
            config = json.load(f)
            connections = config if isinstance(config, list) else config.get("connections", [])
    return df, connections

def reconstruct_3d_sequence(df, triangulator):
    frames = sorted(df['frame'].unique())
    results = {}
    print(f"Reconstructing {len(frames)} frames...")
    for i, frame in enumerate(frames):
        if i % 50 == 0: print(f"Processing frame {frame}...")
        frame_data = df[df['frame'] == frame]
        results[frame] = {}
        for part in df['body_part'].unique():
            part_df = frame_data[frame_data['body_part'] == part]
            if len(part_df) >= 2:
                pt = triangulator.triangulate_n_views(part_df['camera'].tolist(), part_df[['x', 'y']].values.tolist())
                if pt is not None: results[frame][part] = pt
    return results, frames

# =========================================================
#  可視化 (元の構造 + 関節名表示)
# =========================================================
def visualize_skeleton_animation(results, frames, connections, invert_y=True):
    if not frames:
        print("No frames to animate.")
        return

    # キー設定の無効化（前のコードと同じ処理）
    try:
        if 'left' in plt.rcParams['keymap.back']:
            plt.rcParams['keymap.back'].remove('left')
        if 'right' in plt.rcParams['keymap.forward']:
            plt.rcParams['keymap.forward'].remove('right')
    except Exception:
        pass

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # ウィンドウタイトル
    try:
        fig.canvas.manager.set_window_title('Simple Skeleton Animation (ani3)')
    except AttributeError:
        pass

    ax.view_init(elev=10, azim=-80)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if invert_y: ax.invert_yaxis()

    # 範囲設定
    all_points = []
    for f in results:
        for pt in results[f].values(): all_points.append(pt)
    
    if all_points:
        all_points = np.array(all_points)
        center = np.mean(all_points, axis=0)
        max_range = np.max(np.ptp(all_points, axis=0)) / 2.0
        radius = max_range * 1.2 if max_range > 0 else 1.0
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)

    # プロット要素
    scatter = ax.scatter([], [], [], c='r', marker='o', s=30, depthshade=False)
    lines = [ax.plot([], [], [], linewidth=2, color='b')[0] for _ in connections]
    info_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12,
                          bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    
    # ★追加: 関節名ラベル管理用
    joint_texts = {}

    # アニメーション状態
    anim_state = {
        'is_playing': True,
        'frame_idx': 0,
        'last_drawn_idx': -1
    }

    def update(frame_idx):
        if anim_state['is_playing']:
            anim_state['frame_idx'] = (anim_state['frame_idx'] + 1) % len(frames)
        
        current_idx = anim_state['frame_idx']
        # キャッシュ
        if current_idx == anim_state['last_drawn_idx']:
            return scatter, *lines, info_text

        f_num = frames[current_idx]
        pose = results.get(f_num, {})
        info_text.set_text(f"Frame: {f_num}")
        anim_state['last_drawn_idx'] = current_idx

        if not pose:
            scatter._offsets3d = ([], [], [])
            for line in lines: line.set_data([], []); line.set_3d_properties([])
            # データなし時はテキストを隠す
            for t in joint_texts.values(): t.set_text("")
            return scatter, *lines, info_text

        # 点更新
        xs, ys, zs = [], [], []
        for pt in pose.values():
            xs.append(pt[0]); ys.append(pt[1]); zs.append(pt[2])
        scatter._offsets3d = (xs, ys, zs)

        # 線更新
        for line, (s, e) in zip(lines, connections):
            if s in pose and e in pose:
                p1, p2 = pose[s], pose[e]
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line.set_3d_properties([p1[2], p2[2]])
            else:
                line.set_data([], []); line.set_3d_properties([])

        # ★追加: 関節名の更新処理 (安全のためtryで囲む)
        try:
            for part_name, pt in pose.items():
                if part_name not in joint_texts:
                    # 新規作成
                    joint_texts[part_name] = ax.text(pt[0], pt[1], pt[2], part_name, fontsize=9, color='black')
                else:
                    # 位置更新
                    t = joint_texts[part_name]
                    t.set_position((pt[0], pt[1]))
                    t.set_3d_properties(pt[2])
                    t.set_text(part_name)
            
            # データにない部位は隠す
            for part_name, t in joint_texts.items():
                if part_name not in pose:
                    t.set_text("")
        except Exception:
            # 万が一テキスト描画でエラーが出てもアニメーションを止めない
            pass

        return scatter, *lines, info_text

    def on_key(event):
        if event.key == ' ':
            anim_state['is_playing'] = not anim_state['is_playing']
        
        if not anim_state['is_playing']:
            if event.key == 'right':
                anim_state['frame_idx'] = (anim_state['frame_idx'] + 1) % len(frames)
                update(0) # 強制更新
                fig.canvas.draw_idle()
            elif event.key == 'left':
                anim_state['frame_idx'] = (anim_state['frame_idx'] - 1 + len(frames)) % len(frames)
                update(0) # 強制更新
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    # アニメーション開始
    ani = animation.FuncAnimation(
        fig, update, frames=None, interval=ANIMATION_INTERVAL, 
        blit=False, cache_frame_data=False
    )
    
    # 最後に plt.show() を呼ぶ (前のコードと同じ配置)
    plt.show()

if __name__ == "__main__":
    triangulator = MultiViewTriangulator(MATRICES_DIR)
    df, connections = load_data()
    
    # 1. 3D復元
    results_raw, frames = reconstruct_3d_sequence(df, triangulator)
    
    # 2. スムージング処理 (ここを追加！)
    # OneEuroFilterなどの定義コードを上に貼っておく必要があります
    results_smooth, frames = apply_temporal_smoothing(results_raw, frames)
    
    # 3. 可視化 (results_smooth を渡す)
    visualize_skeleton_animation(results_smooth, frames, connections, invert_y=INVERT_Y_AXIS)