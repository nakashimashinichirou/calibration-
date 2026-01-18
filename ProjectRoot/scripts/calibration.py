# %%
import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import os
import shutil
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count
import heapq
import time
from scipy.optimize import least_squares

# =========================================================
#  設定エリア
# =========================================================

DATA_DIR = "../data"
INTRINSICS_DIR = "../calibration_data"
OUTPUT_DIR = "../system_matrices"
DEBUG_IMG_DIR = os.path.join(OUTPUT_DIR, "debug_frames")

# --- ChArUcoボード設定 ---
SQUARE_LENGTH = 90.0
MARKER_LENGTH = 70.0
CHARUCO_SIZE = (6, 9)
ARUCO_DICT_ID = aruco.DICT_4X4_50 

IMAGE_FORMAT = 'jpg'

# ペア計算の設定
NUM_SAMPLES_PAIR = 6000     
MIN_COMMON_POINTS = 6       
DROP_WORST_PERCENT = 20
ITERATION_COUNT = 3          

# 単眼チェックの閾値（ピクセル）
SINGLE_REPROJ_THRESH = 4.0   

# 角度フィルタリング閾値（度）
MIN_BOARD_ANGLE = 0       

# キャリブレーション実行に必要な最低フレーム数
MIN_FRAMES_TO_CALIB = 5

# 基準となるカメラ
ROOT_CAMERA = "Pixel"

CAMERA_NAMES = ["Pixel", "Oppo", "Oppo1", "Oppo2"]

CAMERA_CONFIGS = {
    "Pixel": { "dir": os.path.join(DATA_DIR, "pixel"), "prefix": "pixel_", "mtx": "PIXEL_mtx.npy", "dist": "PIXEL_dist.npy" },
    "Oppo":  { "dir": os.path.join(DATA_DIR, "oppo"),  "prefix": "oppo_",  "mtx": "OPPO_mtx.npy",  "dist": "OPPO_dist.npy" },
    "Oppo1": { "dir": os.path.join(DATA_DIR, "oppo1"), "prefix": "oppo1_", "mtx": "OPPO1_mtx.npy", "dist": "OPPO1_dist.npy" },
    "Oppo2": { "dir": os.path.join(DATA_DIR, "oppo2"), "prefix": "oppo2_", "mtx": "OPPO2_mtx.npy", "dist": "OPPO2_dist.npy" }
}

PAIR_CONFIG_LIST = [
    { "name": "Pixel-Oppo",   "cam_L": "Pixel", "cam_R": "Oppo" },
    { "name": "Pixel-Oppo1",  "cam_L": "Pixel", "cam_R": "Oppo1" },
    { "name": "Pixel-Oppo2",  "cam_L": "Pixel", "cam_R": "Oppo2" },
    { "name": "Oppo-Oppo1",   "cam_L": "Oppo",  "cam_R": "Oppo1" },
    { "name": "Oppo-Oppo2",   "cam_L": "Oppo",  "cam_R": "Oppo2" },
    { "name": "Oppo1-Oppo2",  "cam_L": "Oppo1", "cam_R": "Oppo2" },
]

# =========================================================
#  共通関数
# =========================================================

try:
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = aruco.CharucoBoard(
        size=CHARUCO_SIZE, 
        squareLength=SQUARE_LENGTH, 
        markerLength=MARKER_LENGTH, 
        dictionary=aruco_dict
    )
except AttributeError:
    aruco_dict = aruco.Dictionary_get(ARUCO_DICT_ID)
    board = aruco.CharucoBoard_create(
        squaresX=CHARUCO_SIZE[0], 
        squaresY=CHARUCO_SIZE[1], 
        squareLength=SQUARE_LENGTH, 
        markerLength=MARKER_LENGTH, 
        dictionary=aruco_dict
    )

def load_intrinsics(cam_name):
    conf = CAMERA_CONFIGS[cam_name]
    m_path = os.path.join(INTRINSICS_DIR, conf["mtx"])
    d_path = os.path.join(INTRINSICS_DIR, conf["dist"])
    if not os.path.exists(m_path) or not os.path.exists(d_path):
        return None, None
    return np.load(m_path), np.load(d_path)

def detect_charuco(img):
    if img is None: return None, None
    
    params = aruco.DetectorParameters() if hasattr(aruco, 'DetectorParameters') else aruco.DetectorParameters_create()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if hasattr(aruco, 'CharucoDetector'):
        detector = aruco.CharucoDetector(board, detectorParams=params)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(img)
    else:
        corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=params)
        if len(corners) > 0:
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, img, board
            )
        else:
            return None, None

    if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
        return charuco_corners, charuco_ids
    return None, None

def calculate_reprojection_error_single(objpoints, imgpoints, mtx, dist):
    if mtx is None: return 0.0
    ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, mtx, dist)
    if not ret: return 9999.0
    imgpoints2, _ = cv2.projectPoints(objpoints, rvec, tvec, mtx, dist)
    return cv2.norm(imgpoints, imgpoints2, cv2.NORM_L2) / len(imgpoints)

def calculate_board_angle(objpoints, imgpoints, mtx, dist):
    if mtx is None: return 90.0 
    ret, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, mtx, dist)
    if not ret: return 0.0
    R, _ = cv2.Rodrigues(rvec)
    normal_vector_z = R[:, 2] 
    val = np.clip(np.abs(normal_vector_z[2]), 0.0, 1.0)
    angle_rad = np.arccos(val)
    return np.degrees(angle_rad)

# =========================================================
#  Worker 1: 画像解析タスク
# =========================================================

def process_image_pair_task(args):
    path_left, path_right, cam_L_name, cam_R_name, pair_key = args

    img_left = cv2.imread(path_left)
    img_right = cv2.imread(path_right)
    if img_left is None or img_right is None: return None

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    corners_L, ids_L = detect_charuco(gray_left)
    corners_R, ids_R = detect_charuco(gray_right)

    if corners_L is None or corners_R is None: return None

    ids_L_flat = ids_L.flatten()
    ids_R_flat = ids_R.flatten()
    common_ids = np.intersect1d(ids_L_flat, ids_R_flat)

    if len(common_ids) < MIN_COMMON_POINTS:
        return None

    points_L = []
    points_R = []
    obj_points_frame = []
    all_board_corners = board.getChessboardCorners()

    for cid in common_ids:
        idx_l = np.where(ids_L_flat == cid)[0][0]
        points_L.append(corners_L[idx_l])
        idx_r = np.where(ids_R_flat == cid)[0][0]
        points_R.append(corners_R[idx_r])
        obj_points_frame.append(all_board_corners[cid])

    points_L = np.array(points_L, dtype=np.float32)
    points_R = np.array(points_R, dtype=np.float32)
    obj_points_frame = np.array(obj_points_frame, dtype=np.float32)

    mtx_L, dist_L = load_intrinsics(cam_L_name)
    mtx_R, dist_R = load_intrinsics(cam_R_name)
    
    if mtx_L is not None:
        if calculate_reprojection_error_single(obj_points_frame, points_L, mtx_L, dist_L) > SINGLE_REPROJ_THRESH:
            return None
    if mtx_R is not None:
        if calculate_reprojection_error_single(obj_points_frame, points_R, mtx_R, dist_R) > SINGLE_REPROJ_THRESH:
            return None

    if mtx_L is not None:
        angle_L = calculate_board_angle(obj_points_frame, points_L, mtx_L, dist_L)
        if angle_L < MIN_BOARD_ANGLE: 
            return None
    if mtx_R is not None:
        angle_R = calculate_board_angle(obj_points_frame, points_R, mtx_R, dist_R)
        if angle_R < MIN_BOARD_ANGLE:
            return None

    common_ids_array = np.array(common_ids)[:, np.newaxis]

    return {
        "pair_key": pair_key,
        "obj": obj_points_frame,
        "imgL": points_L,
        "imgR": points_R,
        "idsL": common_ids_array,
        "idsR": common_ids_array,
        "shape": gray_left.shape[::-1],
        "paths": (path_left, path_right)
    }

# =========================================================
#  Worker 2: キャリブレーション計算タスク
# =========================================================

def calc_stereo_error_per_frame(objp, imgp_L, imgp_R, mtx_L, dist_L, mtx_R, dist_R, R, T):
    ret, rvec_L, tvec_L = cv2.solvePnP(objp, imgp_L, mtx_L, dist_L)
    if not ret: return 9999.0
    
    R_L_mat, _ = cv2.Rodrigues(rvec_L)
    R_R_mat = R @ R_L_mat
    tvec_R_est = R @ tvec_L + T
    
    rvec_R_est, _ = cv2.Rodrigues(R_R_mat)
    imgp_R_est, _ = cv2.projectPoints(objp, rvec_R_est, tvec_R_est, mtx_R, dist_R)
    
    return cv2.norm(imgp_R, imgp_R_est, cv2.NORM_L2) / len(imgp_R)

def run_pair_calibration_task(args):
    pair_name, cam_L, cam_R, data_list, image_size = args
    
    mtx_left, dist_left = load_intrinsics(cam_L)
    mtx_right, dist_right = load_intrinsics(cam_R)
    
    current_data = data_list
    final_rms = 9999.0
    final_R, final_T = None, None
    log_messages = []

    log_messages.append(f"[{pair_name}] 開始: {len(current_data)} フレーム")

    for loop_i in range(ITERATION_COUNT + 1):
        n_frames = len(current_data)
        if n_frames < MIN_FRAMES_TO_CALIB:
            log_messages.append(f"[{pair_name}] フレーム不足で中断 (残 {n_frames})")
            return log_messages, None
        
        objpoints = [d["obj"] for d in current_data]
        imgpoints_left = [d["imgL"] for d in current_data]
        imgpoints_right = [d["imgR"] for d in current_data]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        flags = cv2.CALIB_FIX_INTRINSIC
        
        try:
            ret_val, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right,
                mtx_left, dist_left, mtx_right, dist_right,
                image_size, criteria=criteria, flags=flags
            )
        except cv2.error as e:
            log_messages.append(f"[{pair_name}] OpenCV計算エラー: {e}")
            return log_messages, None

        final_rms = ret_val
        final_R, final_T = R, T
        
        if loop_i == ITERATION_COUNT:
            break
        
        errors = []
        for i in range(n_frames):
            err = calc_stereo_error_per_frame(
                objpoints[i], imgpoints_left[i], imgpoints_right[i],
                mtx_left, dist_left, mtx_right, dist_right, R, T
            )
            errors.append(err)
        
        sorted_indices = np.argsort(errors)
        drop_count = int(n_frames * (DROP_WORST_PERCENT / 100.0))
        keep_count = n_frames - drop_count
        
        keep_indices = sorted_indices[:keep_count]
        current_data = [current_data[i] for i in keep_indices]
        
        log_messages.append(f"  Loop {loop_i}: RMS={ret_val:.4f}, 残り{keep_count}枚")

    result_data = None
    if final_rms < 100.0:
        log_messages.append(f"✅ [{pair_name}] 完了 RMS={final_rms:.4f} (枚数: {len(current_data)})")
        
        result_data = {
            "cam_L": cam_L, "cam_R": cam_R,
            "rms": final_rms, "R": final_R, "T": final_T
        }

        save_dir = os.path.join(DEBUG_IMG_DIR, pair_name)
        if os.path.exists(save_dir): shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        for idx, d in enumerate(current_data):
            try:
                frame_err = calc_stereo_error_per_frame(
                    d["obj"], d["imgL"], d["imgR"],
                    mtx_left, dist_left, mtx_right, dist_right, final_R, final_T
                )

                path_L, path_R = d["paths"]
                imgL = cv2.imread(path_L)
                imgR = cv2.imread(path_R)
                
                if imgL is not None and imgR is not None:
                    aruco.drawDetectedCornersCharuco(imgL, d["imgL"], d["idsL"], (0, 255, 0))
                    aruco.drawDetectedCornersCharuco(imgR, d["imgR"], d["idsR"], (0, 255, 0))
                    
                    concat = np.hstack((imgL, imgR))
                    
                    cv2.putText(concat, f"Err: {frame_err:.2f} px", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

                    fname = f"valid_{idx:03d}_err{frame_err:.2f}.jpg"
                    cv2.imwrite(os.path.join(save_dir, fname), concat)
            except Exception:
                pass

    return log_messages, result_data

# =========================================================
#  全体最適化 (Pose Graph Optimization)
# =========================================================

def mat2params(R, T):
    """回転行列と並進ベクトルを、6次元ベクトル(rvec, tvec)に変換"""
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate((rvec.flatten(), T.flatten()))

def params2mat(params):
    """6次元ベクトルを回転行列と並進ベクトルに変換"""
    rvec = params[:3]
    tvec = params[3:]
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1)

def optimization_func(x, edges, cam_map, root_idx):
    """
    最小二乗法のための誤差関数
    各エッジ(計測された相対位置)と、現在のグローバル位置から計算される相対位置の差を返す
    """
    residuals = []
    
    # x配列から各カメラのグローバル行列を復元
    # 6パラメータ * (カメラ数 - 1) 。Rootは単位行列固定。
    
    global_poses = {}
    current_idx = 0
    
    sorted_cams = sorted(cam_map.keys(), key=lambda k: cam_map[k])
    
    for cam_name in sorted_cams:
        idx = cam_map[cam_name]
        if idx == root_idx:
            global_poses[idx] = (np.eye(3), np.zeros((3, 1)))
        else:
            params = x[current_idx : current_idx + 6]
            global_poses[idx] = params2mat(params)
            current_idx += 6
            
    # 全エッジについて誤差計算
    for (name_u, name_v, R_meas, T_meas, weight) in edges:
        idx_u = cam_map[name_u]
        idx_v = cam_map[name_v]
        
        R_u, T_u = global_poses[idx_u]
        R_v, T_v = global_poses[idx_v]
        
        # グローバル位置から予測される相対変換 T_uv_pred = T_u^-1 * T_v
        # T_u^-1 = [R_u^T, -R_u^T * t_u]
        # P_v = T_uv * P_u  => T_uv = P_v * P_u^-1 ... 逆だ
        # ワールド座標系 W から カメラ座標系 C への変換行列を M_c とする (w2c)
        # 観測 T_uv は、C_u座標系から C_v座標系への変換 (u -> v)
        # 点 P_c_u を 点 P_c_v に変換: P_c_v = R_meas * P_c_u + T_meas
        # 一方、グローバルでは: P_c_v = M_v * P_w, P_c_u = M_u * P_w => P_w = M_u^-1 * P_c_u
        # P_c_v = M_v * (M_u^-1 * P_c_u)
        # よって、予測される相対変換は M_v * M_u^-1
        
        # M_u^-1 = [R_u.T, -R_u.T @ T_u]
        # M_v @ M_u^-1
        # R_pred = R_v @ R_u.T
        # T_pred = -R_v @ R_u.T @ T_u + T_v
        
        R_pred = R_v @ R_u.T
        T_pred = T_v - R_pred @ T_u
        
        # 誤差: 観測値 - 予測値
        # 回転の誤差は rvec の差分で近似
        R_err = R_meas @ R_pred.T # R_meas * R_pred^-1 が単位行列に近いほどよい
        rvec_err, _ = cv2.Rodrigues(R_err)
        
        t_err = T_meas - T_pred
        
        # 重みづけ (RMSが小さいほど信頼できる＝重みを大きく)
        w = 1.0 / (weight + 1e-5)
        
        residuals.extend((rvec_err.flatten() * w).tolist())
        residuals.extend((t_err.flatten() * w).tolist())
        
    return np.array(residuals)

def run_global_optimization(transforms_init, graph_data):
    print("\n" + "="*60)
    print("全体最適化 (Pose Graph Optimization)")
    print("="*60)
    
    # マッピング作成
    cam_map = {name: i for i, name in enumerate(CAMERA_NAMES)}
    root_idx = cam_map[ROOT_CAMERA]
    
    # 1. 初期値ベクトル x0 の作成
    x0 = []
    sorted_cams = sorted(cam_map.keys(), key=lambda k: cam_map[k])
    
    for cam_name in sorted_cams:
        idx = cam_map[cam_name]
        if idx == root_idx: continue # Rootは最適化しない
        
        # Dijkstraで求めた初期値 (World -> Cam)
        M_w2c = transforms_init.get(cam_name, np.eye(4))
        R_init = M_w2c[:3, :3]
        T_init = M_w2c[:3, 3].reshape(3, 1)
        x0.extend(mat2params(R_init, T_init))
        
    x0 = np.array(x0)
    
    # 2. エッジリストの作成 (RMSが大きいエッジも、情報として使う)
    edges = []
    used_pairs = set()
    
    for cam_u, connections in graph_data.items():
        for (cam_v, rms, R, T) in connections:
            pair_key = tuple(sorted((cam_u, cam_v)))
            if pair_key in used_pairs: continue
            
            # RMSがあまりに酷いもの(例えば20px以上)は除外してもいいが、
            # ここでは重み付けで対処するので全て入れる
            edges.append((cam_u, cam_v, R, T, rms))
            used_pairs.add(pair_key)
            print(f"  Constraint: {cam_u} <-> {cam_v} (RMS={rms:.2f})")
            
    # 3. 最適化実行
    print(f"\n最適化開始 (変数の数: {len(x0)})...")
    res = least_squares(optimization_func, x0, args=(edges, cam_map, root_idx), verbose=1)
    
    print(f"完了. Cost: {res.cost:.4f}")
    
    # 4. 結果の展開
    optimized_transforms = {}
    current_idx = 0
    
    for cam_name in sorted_cams:
        idx = cam_map[cam_name]
        if idx == root_idx:
            optimized_transforms[cam_name] = np.eye(4)
        else:
            params = res.x[current_idx : current_idx + 6]
            current_idx += 6
            R_opt, T_opt = params2mat(params)
            
            M = np.eye(4)
            M[:3, :3] = R_opt
            M[:3, 3] = T_opt.flatten()
            optimized_transforms[cam_name] = M
            
    return optimized_transforms

# =========================================================
#  パイプライン実行
# =========================================================

def run_pipeline():
    print("\n" + "="*60)
    print(f"全体最適化対応キャリブレーション (CPU: {cpu_count()} cores)")
    print("="*60)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    os.makedirs(DEBUG_IMG_DIR)

    # 1. 画像検索
    print("画像ペアを検索中...")
    all_image_tasks = []
    
    for config in PAIR_CONFIG_LIST:
        pair_name = config["name"]
        cam_L = config["cam_L"]
        cam_R = config["cam_R"]
        conf_L = CAMERA_CONFIGS[cam_L]
        conf_R = CAMERA_CONFIGS[cam_R]

        search_pattern = os.path.join(conf_L["dir"], f'{conf_L["prefix"]}*.{IMAGE_FORMAT}')
        paths_left = sorted(glob.glob(search_pattern))
        
        pairs = []
        for pL in paths_left:
            base = os.path.basename(pL)
            num = base.replace(conf_L["prefix"], '').replace(f'.{IMAGE_FORMAT}', '')
            pR = os.path.join(conf_R["dir"], f'{conf_R["prefix"]}{num}.{IMAGE_FORMAT}')
            if os.path.exists(pR):
                pairs.append((pL, pR))

        if len(pairs) > NUM_SAMPLES_PAIR:
            pairs = random.sample(pairs, NUM_SAMPLES_PAIR)
        
        for pL, pR in pairs:
            all_image_tasks.append( (pL, pR, cam_L, cam_R, pair_name) )

    print(f"処理対象候補: {len(all_image_tasks)} ペア")
    
    # 2. 並列画像解析
    print("\n[Phase 1] 画像解析と角度フィルタリング...")
    grouped_results = {cfg["name"]: [] for cfg in PAIR_CONFIG_LIST}
    pair_image_sizes = {}

    start_time = time.time()
    valid_count = 0
    with Pool(processes=cpu_count()) as pool:
        for res in pool.imap_unordered(process_image_pair_task, all_image_tasks, chunksize=10):
            if res is not None:
                key = res["pair_key"]
                grouped_results[key].append(res)
                if key not in pair_image_sizes:
                    pair_image_sizes[key] = res["shape"]
                valid_count += 1
    
    print(f"解析完了: {time.time() - start_time:.1f} 秒 (有効ペア: {valid_count} / {len(all_image_tasks)})")

    # 3. キャリブレーション (エッジ作成)
    print("\n[Phase 2] ステレオキャリブレーション (エッジ構築)...")
    
    calib_tasks = []
    for pair_name, data_list in grouped_results.items():
        if len(data_list) < MIN_FRAMES_TO_CALIB:
            print(f"[{pair_name}] スキップ: データ不足 ({len(data_list)} frames)")
            continue
        
        cam_L = [c for c in PAIR_CONFIG_LIST if c["name"]==pair_name][0]["cam_L"]
        cam_R = [c for c in PAIR_CONFIG_LIST if c["name"]==pair_name][0]["cam_R"]
        img_size = pair_image_sizes[pair_name]
        
        calib_tasks.append( (pair_name, cam_L, cam_R, data_list, img_size) )

    graph_data = {name: [] for name in CAMERA_NAMES}
    
    with Pool(processes=min(len(calib_tasks), cpu_count())) as pool:
        results = pool.map(run_pair_calibration_task, calib_tasks)
        
        for logs, data in results:
            for log in logs: print("  " + log)
            if data:
                cL, cR = data["cam_L"], data["cam_R"]
                R, T, rms = data["R"], data["T"], data["rms"]
                graph_data[cL].append( (cR, rms, R, T) )
                
                R_inv = R.T
                T_inv = -R_inv @ T
                graph_data[cR].append( (cL, rms, R_inv, T_inv) )

    return graph_data

def calculate_initial_guess(graph):
    """Dijkstra法で初期値を計算し、経路も記録する"""
    print("\n" + "="*60)
    print("初期値計算 (Dijkstra)")
    print("="*60)
    
    transforms_w2c = {}
    transforms_w2c[ROOT_CAMERA] = np.eye(4)
    
    # (cost, current_node, path_string)
    pq = [(0, ROOT_CAMERA, ROOT_CAMERA)]
    visited = {}
    path_info = {} # node -> path_string

    while pq:
        cost, u, path_str = heapq.heappop(pq)
        
        if u in visited and visited[u] < cost: continue
        visited[u] = cost
        path_info[u] = path_str
        
        if u in graph:
            for v, weight, R, T in graph[u]:
                new_cost = cost + weight
                if v not in visited or new_cost < visited.get(v, float('inf')):
                    # 座標計算
                    M_w2c_u = transforms_w2c[u]
                    M_rel = np.eye(4)
                    M_rel[:3, :3] = R
                    M_rel[:3, 3] = T.flatten()
                    transforms_w2c[v] = M_rel @ M_w2c_u
                    
                    new_path = path_str + " -> " + v
                    heapq.heappush(pq, (new_cost, v, new_path))
                    
    # 結果表示
    for cam in CAMERA_NAMES:
        if cam in path_info:
            print(f"-> {cam}: {path_info[cam]} (初期累積RMS: {visited[cam]:.2f})")
            
    return transforms_w2c

def save_and_visualize(transforms, suffix=""):
    poses_c2w = {}
    print(f"\n=== カメラ位置 (World) {suffix} ===")
    
    for name, M_w2c in transforms.items():
        np.save(os.path.join(OUTPUT_DIR, f"{name}_extrinsic_world2cam{suffix}.npy"), M_w2c)
        M_c2w = np.linalg.inv(M_w2c)
        poses_c2w[name] = M_c2w
        np.save(os.path.join(OUTPUT_DIR, f"{name}_pose_cam2world{suffix}.npy"), M_c2w)
        print(f"[{name}] {M_c2w[:3, 3]}")
        
        cam_conf = CAMERA_CONFIGS[name]
        mtx_path = os.path.join(INTRINSICS_DIR, cam_conf["mtx"])
        if os.path.exists(mtx_path):
             K = np.load(mtx_path)
             P = K @ M_w2c[:3, :]
             np.save(os.path.join(OUTPUT_DIR, f"{name}_projection{suffix}.npy"), P)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(CAMERA_NAMES)}
    
    all_pos = []
    for name, M_c2w in poses_c2w.items():
        pos = M_c2w[:3, 3]
        all_pos.append(pos)
        c = color_map.get(name, 'black')
        ax.scatter(pos[0], pos[1], pos[2], c=c, marker='o', s=100, label=name)
        ax.text(pos[0], pos[1], pos[2], f"  {name}", color=c)
        vec = M_c2w[:3, :3] @ np.array([0, 0, 1]) * (MARKER_LENGTH * 30)  #重要！！マーカーの矢印の長さは長く！！
        ax.quiver(pos[0], pos[1], pos[2], vec[0], vec[1], vec[2], color=c, alpha=0.5)

    if all_pos:
        all_pos = np.array(all_pos)
        center = np.mean(all_pos, axis=0)
        max_range = (np.max(all_pos, axis=0) - np.min(all_pos, axis=0)).max() / 2.0 + 500
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    plt.title(f"Camera Poses {suffix}")
    plt.show()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 1. ペアごとのキャリブレーション
    graph_data = run_pipeline()
    
    if graph_data:
        # 2. 初期値推定 (Dijkstra)
        initial_transforms = calculate_initial_guess(graph_data)
        
        # 3. 全体最適化 (Pose Graph Optimization)
        # 全てのリンクを使って、矛盾が最小になるように位置を微調整する
        final_transforms = run_global_optimization(initial_transforms, graph_data)
        
        # 4. 結果保存
        save_and_visualize(final_transforms)

    print("\n全工程完了。")
# %%