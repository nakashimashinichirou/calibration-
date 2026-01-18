import cv2
import sys
import json
import os
import csv
import datetime
import numpy as np
import tkinter as tk
from tkinter import simpledialog, ttk, messagebox
from PIL import Image, ImageDraw, ImageFont 
import platform
from concurrent.futures import ThreadPoolExecutor 

# --- 設定 ---
SYNC_CONFIG_FILE = "../data/sync_config.json"
OUTPUT_CSV_FILE = "../data/2d_points.csv"
STATE_FILE = "../data/session_state.json" 

SCALING_FACTOR = 1.0 
CAM_W, CAM_H = 1920, 1080
MAX_WORKERS = 8 

BODY_PARTS = [
    "右肩", "右肘", "右手首",
    "左肩", "左肘", "左手首",
    "右腰", "右膝", "右足首",
    "左腰", "左膝", "左足首"
]

class MultiCamTrackerApp:
    def __init__(self):
        self.load_config()
        self.window_name = "Multi-Camera Motion Capture Dashboard"
        self.fullscreen = False
        
        self.font = self.load_japanese_font()
        
        # 状態変数
        self.global_frame = 0
        self.total_frames = 10000
        self.start_sync = 0
        self.end_sync = 100
        self.is_playing = False
        self.tracking_data = {} 
        self.unsaved_changes = False
        self.show_status_panel = True # パネル表示フラグ
        
        # UI操作用変数
        self.mouse_x, self.mouse_y = 0, 0
        self.roi_drag_start = None
        self.roi_drag_end = None
        self.selecting_roi = False
        self.is_dragging = False
        self.manual_targeting_part = None 
        
        # パネル移動用
        self.panel_x = 20
        self.panel_y = 230
        self.panel_w = 400
        self.panel_h = 30 + len(BODY_PARTS) * 25
        self.is_dragging_panel = False
        self.panel_drag_offset = (0, 0)

        # 描画キャッシュ
        self.cached_combined_image = None
        self.last_rendered_frame = -1
        self.force_redraw = False
        
        # 動画ロード
        self.caps = {}
        self.offsets = {}
        self.camera_names = sorted(list(self.sync_data.keys()))
        self.prepare_videos()
        
        # データロード
        self.load_from_csv()
        self.load_state() 

    def load_config(self):
        if not os.path.exists(SYNC_CONFIG_FILE):
            print(f"Error: {SYNC_CONFIG_FILE} not found.")
            sys.exit()
        with open(SYNC_CONFIG_FILE, 'r') as f:
            self.sync_data = json.load(f)

    def load_japanese_font(self):
        font_path = None
        system = platform.system()
        candidates = []
        if system == "Windows":
            candidates = ["C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/msgothic.ttc"]
        elif system == "Darwin": 
            candidates = ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/Library/Fonts/Arial Unicode.ttf"]
        else:
            candidates = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
        
        for path in candidates:
            if os.path.exists(path):
                font_path = path
                break
        try:
            return ImageFont.truetype(font_path, 24) if font_path else ImageFont.load_default()
        except:
            return ImageFont.load_default()

    def prepare_videos(self):
        min_duration = float('inf')
        for name in self.camera_names:
            path = self.sync_data[name]['video_path']
            offset = int(self.sync_data[name]['offset_frame'])
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Failed to open {path}")
                sys.exit()
            self.caps[name] = cap
            self.offsets[name] = offset
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count - offset
            if duration < min_duration:
                min_duration = duration

        self.total_frames = max(100, int(min_duration))
        print(f"Total Sync Duration: {self.total_frames} frames")

    def load_from_csv(self):
        if not os.path.exists(OUTPUT_CSV_FILE):
            return
        print(f"Loading existing data from {OUTPUT_CSV_FILE}...")
        try:
            with open(OUTPUT_CSV_FILE, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                count = 0
                for row in reader:
                    if len(row) < 5: continue
                    try:
                        frame = int(row[0])
                        cam = row[1]
                        part = row[2]
                        x = float(row[3])
                        y = float(row[4])
                        if part not in self.tracking_data: self.tracking_data[part] = {}
                        if frame not in self.tracking_data[part]: self.tracking_data[part][frame] = {}
                        self.tracking_data[part][frame][cam] = (x, y)
                        count += 1
                    except ValueError:
                        continue
            print(f"Loaded {count} points.")
            self.unsaved_changes = False 
        except Exception as e:
            print(f"Error loading CSV: {e}")

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.start_sync = data.get("start_sync", 0)
                    self.end_sync = data.get("end_sync", 100)
                    print(f"Restored Session: Start={self.start_sync}, End={self.end_sync}")
                    self.global_frame = self.start_sync
            except Exception as e:
                print(f"Failed to load session state: {e}")

    def save_state(self):
        data = {
            "start_sync": self.start_sync,
            "end_sync": self.end_sync
        }
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save session state: {e}")        

    # --- UI操作コールバック ---
    def mouse_cb(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        
        # --- 1. パネルドラッグ移動 ---
        if self.show_status_panel:
            if event == cv2.EVENT_LBUTTONDOWN:
                if (self.panel_x <= x <= self.panel_x + self.panel_w and 
                    self.panel_y <= y <= self.panel_y + 40):
                    self.is_dragging_panel = True
                    self.panel_drag_offset = (x - self.panel_x, y - self.panel_y)
                    self.force_redraw = True
                    return 
            elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging_panel:
                self.panel_x = x - self.panel_drag_offset[0]
                self.panel_y = y - self.panel_drag_offset[1]
                self.force_redraw = True
                return
            elif event == cv2.EVENT_LBUTTONUP and self.is_dragging_panel:
                self.is_dragging_panel = False
                self.force_redraw = True
                return

        real_x = x / SCALING_FACTOR
        real_y = y / SCALING_FACTOR

        # --- 2. マニュアル修正モード ---
        if self.manual_targeting_part:
            if event == cv2.EVENT_LBUTTONDOWN:
                col = int(real_x // CAM_W)
                row = int(real_y // CAM_H)
                
                if 0 <= col <= 1 and 0 <= row <= 1:
                    cam_idx = row * 2 + col
                    if cam_idx < len(self.camera_names):
                        cam_name = self.camera_names[cam_idx]
                        
                        # A. [Ctrl] + Click -> 未来すべて削除 (Cancel Tracking)
                        if flags & cv2.EVENT_FLAG_CTRLKEY:
                            print(f"Manual: DELETE FUTURE for {self.manual_targeting_part} on {cam_name} (Frame {self.global_frame} -> End)")
                            for f in range(self.global_frame, self.total_frames):
                                self.save_point(self.manual_targeting_part, f, cam_name, None, None)

                        # B. [Shift] + Click -> 現在の点のみ削除 (Delete)
                        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                            print(f"Manual: DELETE point for {self.manual_targeting_part} on {cam_name} at {self.global_frame}")
                            self.save_point(self.manual_targeting_part, self.global_frame, cam_name, None, None)
                        
                        # C. Click -> 点を登録 (Set Point)
                        else:
                            local_x = real_x % CAM_W
                            local_y = real_y % CAM_H
                            print(f"Manual: SET point for {self.manual_targeting_part} on {cam_name}")
                            self.save_point(self.manual_targeting_part, self.global_frame, cam_name, local_x, local_y)
                        
                        self.force_redraw = True
                        return

        # --- 3. ROI選択モード ---
        if self.selecting_roi:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_drag_start = (x, y)
                self.roi_drag_end = (x, y)
                self.is_dragging = True
                self.force_redraw = True
            elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
                self.roi_drag_end = (x, y)
                self.force_redraw = True
            elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
                self.roi_drag_end = (x, y)
                self.is_dragging = False
                self.force_redraw = True

        # --- 4. フルスクリーン切り替え ---
        if event == cv2.EVENT_LBUTTONDOWN and not self.manual_targeting_part and not self.selecting_roi:
            w = int(CAM_W * 2 * SCALING_FACTOR)
            if x > w - 60 and y < 60:
                self.set_fullscreen(not self.fullscreen)
        self.mouse_x, self.mouse_y = x, y
        
        # --- 1. パネルドラッグ移動の判定 ---
        if self.show_status_panel:
            if event == cv2.EVENT_LBUTTONDOWN:
                if (self.panel_x <= x <= self.panel_x + self.panel_w and 
                    self.panel_y <= y <= self.panel_y + 40): # ヘッダー部分のみ掴める
                    self.is_dragging_panel = True
                    self.panel_drag_offset = (x - self.panel_x, y - self.panel_y)
                    self.force_redraw = True
                    return # パネル操作時は他のクリックを無効化
            elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging_panel:
                self.panel_x = x - self.panel_drag_offset[0]
                self.panel_y = y - self.panel_drag_offset[1]
                self.force_redraw = True
                return
            elif event == cv2.EVENT_LBUTTONUP and self.is_dragging_panel:
                self.is_dragging_panel = False
                self.force_redraw = True
                return

        # 映像内座標への変換
        real_x = x / SCALING_FACTOR
        real_y = y / SCALING_FACTOR

        # --- 2. マニュアル修正モード ---
        if self.manual_targeting_part:
            # 右クリック: 削除処理
            if event == cv2.EVENT_RBUTTONDOWN:
                col = int(real_x // CAM_W)
                row = int(real_y // CAM_H)
                if 0 <= col <= 1 and 0 <= row <= 1:
                    cam_idx = row * 2 + col
                    if cam_idx < len(self.camera_names):
                        cam_name = self.camera_names[cam_idx]
                        
                        # Shift + 右クリック: 「ここから最後まで」削除 (トラッキングキャンセル)
                        if flags & cv2.EVENT_FLAG_SHIFTKEY:
                            print(f"Manual: CLEAR TRACKING for {self.manual_targeting_part} on {cam_name} from {self.global_frame} to END.")
                            for f in range(self.global_frame, self.total_frames): # total_framesまで一気に消す
                                self.save_point(self.manual_targeting_part, f, cam_name, None, None)
                        
                        # 通常右クリック: 「このフレームだけ」削除
                        else:
                            print(f"Manual: Set None for {self.manual_targeting_part} on {cam_name} at {self.global_frame}")
                            self.save_point(self.manual_targeting_part, self.global_frame, cam_name, None, None)
                        
                        self.force_redraw = True
                        return

            # 左クリック: 点の登録
            elif event == cv2.EVENT_LBUTTONDOWN:
                col = int(real_x // CAM_W)
                row = int(real_y // CAM_H)
                if 0 <= col <= 1 and 0 <= row <= 1:
                    cam_idx = row * 2 + col
                    if cam_idx < len(self.camera_names):
                        cam_name = self.camera_names[cam_idx]
                        local_x = real_x % CAM_W
                        local_y = real_y % CAM_H
                        print(f"Manual: Set Point for {self.manual_targeting_part} on {cam_name} at {self.global_frame}")
                        self.save_point(self.manual_targeting_part, self.global_frame, cam_name, local_x, local_y)
                        self.force_redraw = True
                        return

        # --- 3. ROI選択モード ---
        if self.selecting_roi:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_drag_start = (x, y)
                self.roi_drag_end = (x, y)
                self.is_dragging = True
                self.force_redraw = True
            elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
                self.roi_drag_end = (x, y)
                self.force_redraw = True
            elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
                self.roi_drag_end = (x, y)
                self.is_dragging = False
                self.force_redraw = True

        # --- 4. フルスクリーン切り替え ---
        if event == cv2.EVENT_LBUTTONDOWN:
            w = int(CAM_W * 2 * SCALING_FACTOR)
            if x > w - 60 and y < 60:
                self.set_fullscreen(not self.fullscreen)

    def set_fullscreen(self, enable):
        self.fullscreen = enable
        prop = cv2.WINDOW_FULLSCREEN if enable else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)

    def on_trackbar(self, val):
        self.global_frame = val
        self.is_playing = False

    def put_text_jp(self, img, text, pos, color=(0, 255, 0)):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((pos[0]+2, pos[1]+2), text, font=self.font, fill=(0,0,0))
        draw.text(pos, text, font=self.font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def get_combined_view(self, draw_ui=True):
        if (not self.is_playing 
            and self.global_frame == self.last_rendered_frame 
            and self.cached_combined_image is not None 
            and not self.force_redraw):
            return self.cached_combined_image

        frames = []
        for i, name in enumerate(self.camera_names):
            cap = self.caps[name]
            target = self.offsets[name] + self.global_frame
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != target:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, (CAM_W, CAM_H))
                # カメラ名表示
                cv2.putText(frame, name, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                cv2.putText(frame, name, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
            else:
                frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            frames.append(frame)

        while len(frames) < 4:
            frames.append(np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8))

        if draw_ui:
            for i, name in enumerate(self.camera_names):
                for part_name, history in self.tracking_data.items():
                    if self.global_frame in history:
                        cam_data = history[self.global_frame].get(name)
                        if cam_data and cam_data[0] is not None:
                            cx, cy = int(cam_data[0]), int(cam_data[1])
                            cv2.circle(frames[i], (cx, cy), 8, (0, 0, 255), -1)
                            frames[i] = self.put_text_jp(frames[i], part_name, (cx+10, cy-10), (0, 255, 0))

        top = cv2.hconcat([frames[0], frames[1]])
        btm = cv2.hconcat([frames[2], frames[3]])
        full = cv2.vconcat([top, btm])

        if SCALING_FACTOR != 1.0:
            h, w = full.shape[:2]
            full = cv2.resize(full, (int(w * SCALING_FACTOR), int(h * SCALING_FACTOR)))

        if draw_ui:
            self.draw_overlay(full)

        self.cached_combined_image = full
        self.last_rendered_frame = self.global_frame
        self.force_redraw = False
        return full

    def draw_overlay(self, img):
        h, w = img.shape[:2]
        
        # 保存ボタンエリア
        btn_color = (100, 100, 100)
        cv2.rectangle(img, (w-60, 10), (w-10, 60), btn_color, -1)
        cv2.rectangle(img, (w-50, 20), (w-20, 50), (255, 255, 255), 2)
        
        # 下部ステータスバー
        cv2.rectangle(img, (0, h-80), (w, h), (0, 0, 0), -1)
        status_text = " [UNSAVED]" if self.unsaved_changes else ""
        
        if self.manual_targeting_part:
            mode_info = f"MANUAL MODE [{self.manual_targeting_part}]: Left=Set | Right=None | Shift+Right=Cancel Future"
            cv2.putText(img, mode_info, (400, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        info = f"Frame: {self.global_frame} | Range: {self.start_sync} - {self.end_sync}{status_text}"
        txt_color = (0, 0, 255) if self.unsaved_changes else (0, 255, 255)
        cv2.putText(img, info, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, txt_color, 2)

        # 操作ガイド（簡略化）
        guide_lines = [
            "[R] Track Part (Select -> Cam -> Go)",
            "[M] Manual Edit (Shift+RClick to Cancel)", 
            "[S/E] Set Start/End Frame",
            "[Space] Play/Pause",
            "[T] Toggle Status Panel",
            "[W] Save CSV", 
            "[Ctrl+Q] Quit"
        ]
        cv2.rectangle(img, (10, 10), (450, 190), (0,0,0), -1)
        for i, line in enumerate(guide_lines):
            cv2.putText(img, line, (20, 35 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # ステータスパネル（可動式）
        if self.show_status_panel:
            self.draw_status_panel(img)

    def draw_status_panel(self, img):
        x, y = self.panel_x, self.panel_y
        w, h = self.panel_w, self.panel_h
        
        # 背景
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # ヘッダー（ドラッグ可能エリア）
        cv2.rectangle(img, (x, y), (x + w, y + 30), (60, 60, 60), -1)
        cv2.putText(img, "Status (Drag Header) [T:Hide]", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 内容
        cv2.putText(img, "Part        | Curr | Range(Start-End)", (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        y_off = y + 75
        for part in BODY_PARTS:
            # 現在フレームのカメラ数
            current_cam_count = 0
            if part in self.tracking_data and self.global_frame in self.tracking_data[part]:
                valid_cams = [c for c, p in self.tracking_data[part][self.global_frame].items() if p[0] is not None]
                current_cam_count = len(valid_cams)
            
            # 範囲チェック
            low_coverage_frames = 0
            if part in self.tracking_data:
                history = self.tracking_data[part]
                for f in range(self.start_sync, self.end_sync + 1):
                    cams_at_f = history.get(f, {})
                    valid_cnt = sum(1 for p in cams_at_f.values() if p[0] is not None)
                    if valid_cnt < 2:
                        low_coverage_frames += 1
            else:
                low_coverage_frames = (self.end_sync - self.start_sync + 1)

            range_ok = (low_coverage_frames == 0)
            col_curr = (0, 255, 0) if current_cam_count >= 2 else (0, 0, 255)
            col_range = (0, 255, 0) if range_ok else (0, 0, 255)

            img = self.put_text_jp(img, part, (x + 10, y_off - 15), (200, 200, 200))
            cv2.putText(img, f"{current_cam_count} cams", (x + 150, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_curr, 1)
            range_msg = "OK" if range_ok else f"Low ({low_coverage_frames}f)"
            cv2.putText(img, range_msg, (x + 250, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_range, 1)
            y_off += 25

    # --- トラッキングロジック ---
    def select_roi_on_image(self, img, title_msg):
            self.selecting_roi = True
            self.roi_drag_start = None
            self.roi_drag_end = None
            self.is_dragging = False
            self.force_redraw = True 
            
            h_img, w_img = img.shape[:2]
            
            while True:
                disp = img.copy()
                cv2.rectangle(disp, (0,0), (w_img, 120), (0,0,0), -1)
                disp = self.put_text_jp(disp, title_msg, (20, 20), (0, 255, 0))
                cv2.putText(disp, "Mouse Drag: Select | [Enter]: Confirm | [Esc]: No Data", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                if self.roi_drag_start and self.roi_drag_end:
                    cv2.rectangle(disp, self.roi_drag_start, self.roi_drag_end, (0, 255, 0), 2)

                if SCALING_FACTOR != 1.0:
                    h_s, w_s = int(h_img*SCALING_FACTOR), int(w_img*SCALING_FACTOR)
                    show_img = cv2.resize(disp, (w_s, h_s))
                else:
                    show_img = disp
                
                cv2.imshow(self.window_name, show_img)
                
                key = cv2.waitKey(10) & 0xFF
                
                # Enter(13) または Space(32)
                if key == 13 or key == 32: 
                    if self.roi_drag_start and self.roi_drag_end:
                        # 修正箇所: ここですべて int() に変換します
                        x1 = int(min(self.roi_drag_start[0], self.roi_drag_end[0]) / SCALING_FACTOR)
                        y1 = int(min(self.roi_drag_start[1], self.roi_drag_end[1]) / SCALING_FACTOR)
                        w_rect = int(abs(self.roi_drag_start[0] - self.roi_drag_end[0]) / SCALING_FACTOR)
                        h_rect = int(abs(self.roi_drag_start[1] - self.roi_drag_end[1]) / SCALING_FACTOR)
                        
                        # 小さすぎる範囲は無視
                        if w_rect > 5 and h_rect > 5:
                            self.selecting_roi = False
                            self.force_redraw = True
                            print(f"ROI Confirmed: {w_rect}x{h_rect}") # 確認用ログ
                            return (x1, y1, w_rect, h_rect)
                
                elif key == 27: # Esc
                    self.selecting_roi = False
                    self.force_redraw = True
                    print("ROI Selection Cancelled (None)")
                    return None

    def get_part_name_panel(self):
        root = tk.Tk()
        root.title("Select Part")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = 400, 350
        root.geometry(f"{w}x{h}+{int(sw/2-w/2)}+{int(sh/2-h/2)}")
        
        tk.Label(root, text="部位を選択してください", font=("Arial", 12)).pack(pady=10)
        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        result = [None]
        def on_click(p_name):
            result[0] = p_name
            root.destroy()
        
        for i, part in enumerate(BODY_PARTS):
            btn = tk.Button(frame, text=part, width=10, height=2, command=lambda p=part: on_click(p))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            
        root.wait_window()
        return result[0]

    def get_target_cameras_panel(self):
        root = tk.Tk()
        root.title("Select Cameras")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = 300, 300
        root.geometry(f"{w}x{h}+{int(sw/2-w/2)}+{int(sh/2-h/2)}")
        
        tk.Label(root, text="処理するカメラを選択してください", font=("Arial", 12)).pack(pady=10)
        vars = []
        for name in self.camera_names:
            var = tk.BooleanVar(value=False)
            vars.append((name, var))
            tk.Checkbutton(root, text=name, variable=var, font=("Arial", 16)).pack(anchor='w', padx=50)

        result = []
        # ★修正2: 引数 (event=None) を追加
        def on_ok(event=None):
            for name, var in vars:
                if var.get(): result.append(name)
            root.destroy()
        
        # ★修正3: エンターキー(Return)を紐付け
        root.bind('<Return>', on_ok)
        
        tk.Button(root, text="OK", command=on_ok, width=10).pack(pady=20)
        root.wait_window()
        return result if result else []

    def update_tracker_task(self, tracker, frame):
        if frame is None: return False, None
        return tracker.update(frame)

    def run_tracking_sequence(self, parts_to_track, start_frame, target_cameras=None):
        if target_cameras is None: target_cameras = self.camera_names
        self.global_frame = start_frame
        cv2.setTrackbarPos("Time", self.window_name, self.global_frame)
        self.force_redraw = True
        
        tasks = [] 
        for part_name in parts_to_track:
            for name in target_cameras:
                cap = self.caps[name]
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.offsets[name] + self.global_frame)
                ret, frame = cap.read()
                if not ret: continue
                
                frame = cv2.resize(frame, (CAM_W, CAM_H))
                bbox = self.select_roi_on_image(frame, f"Set ROI: [{part_name}] ({name})")
                
                if bbox:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    tasks.append({'part': part_name, 'cam': name, 'tracker': tracker})
                    self.save_point(part_name, self.global_frame, name, bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)
                else:
                    print(f"Skipping {name} for {part_name} (None)")

        if not tasks: return

        print(f">>> Tracking from {start_frame} to {self.end_sync}...")
        curr = start_frame + 1
        total_steps = self.end_sync - start_frame
        if total_steps <= 0: return

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            while curr <= self.end_sync:
                if (curr - start_frame) % 5 == 0 or curr == self.end_sync:
                    progress = (curr - start_frame) / total_steps * 100
                    loading_screen = np.zeros((600, 800, 3), dtype=np.uint8)
                    cv2.putText(loading_screen, f"Frame {curr} / {self.end_sync}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(loading_screen, f"Progress: {int(progress)}%", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    cv2.imshow(self.window_name, loading_screen)
                    cv2.waitKey(1)
                
                current_frames = {}
                for name in self.camera_names:
                    cap = self.caps[name]
                    ret, frame = cap.read()
                    current_frames[name] = cv2.resize(frame, (CAM_W, CAM_H)) if ret else None

                futures = []
                for task in tasks:
                    future = executor.submit(self.update_tracker_task, task['tracker'], current_frames.get(task['cam']))
                    futures.append((task, future))
                
                for task, future in futures:
                    ok, box = future.result() 
                    name = task['cam']
                    if ok:
                        self.save_point(task['part'], curr, name, box[0] + box[2]/2, box[1] + box[3]/2)
                    else:
                        self.save_point(task['part'], curr, name, None, None)
                curr += 1
        
        self.global_frame = start_frame
        self.force_redraw = True
        cv2.setTrackbarPos("Time", self.window_name, self.global_frame)

    def save_point(self, part, frame, cam, x, y):
        if part not in self.tracking_data: self.tracking_data[part] = {}
        if frame not in self.tracking_data[part]: self.tracking_data[part][frame] = {}
        self.tracking_data[part][frame][cam] = (x, y)
        self.unsaved_changes = True 

    def save_to_csv(self):
        print(f"Saving to {OUTPUT_CSV_FILE}...")
        flattened_data = []
        for part, history in self.tracking_data.items():
            for frame, cams in history.items():
                for cam, coord in cams.items():
                    if coord[0] is not None:
                        flattened_data.append({
                            "frame": frame, "cam": cam, "part": part, "x": coord[0], "y": coord[1]
                        })
        flattened_data.sort(key=lambda item: (item["frame"], item["part"], item["cam"]))
        
        try:
            with open(OUTPUT_CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["SyncFrame", "Camera", "Part", "X", "Y"])
                for item in flattened_data:
                    writer.writerow([item["frame"], item["cam"], item["part"], item["x"], item["y"]])
            print("Saved.")
            self.save_state()
            self.unsaved_changes = False 
        except Exception as e:
            print(f"Failed to save CSV: {e}")

    def confirm_quit(self):
        if self.unsaved_changes:
            root = tk.Tk(); root.withdraw()
            ans = messagebox.askyesnocancel("Unsaved Changes", "変更が保存されていません。保存して終了しますか？")
            root.destroy()
            if ans is True: self.save_to_csv(); return True
            elif ans is False: return True
            else: return False
        return True

    def run(self):
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Time", self.window_name, 0, self.total_frames, self.on_trackbar)
            cv2.setTrackbarPos("Time", self.window_name, self.global_frame)
            cv2.setMouseCallback(self.window_name, self.mouse_cb)
            
            while True:
                img = self.get_combined_view()
                cv2.imshow(self.window_name, img)
                
                delay = 1 if self.is_playing else 30
                key = cv2.waitKey(delay) & 0xFF
                
                if key == ord('w'): 
                    self.save_to_csv()
                    self.force_redraw = True
                
                elif key == 17: # Ctrl + Q
                    if self.confirm_quit(): break
                
                elif key == 27: # Esc: マニュアルモード解除 / キャンセル
                    if self.manual_targeting_part:
                        self.manual_targeting_part = None
                        print("Exited Manual Mode")
                        self.force_redraw = True
                
                elif key == 32: # Space
                    self.is_playing = not self.is_playing
                
                elif key == 81 or key == 2424832: # Left
                    self.is_playing = False
                    if self.global_frame > 0:
                        self.global_frame -= 1
                        cv2.setTrackbarPos("Time", self.window_name, self.global_frame)
                        self.force_redraw = True
                elif key == 83 or key == 2555904: # Right
                    self.is_playing = False
                    if self.global_frame < self.total_frames:
                        self.global_frame += 1
                        cv2.setTrackbarPos("Time", self.window_name, self.global_frame)
                        self.force_redraw = True
                
                elif key == ord('s'): 
                    self.start_sync = self.global_frame
                    print(f"Start: {self.start_sync}")
                    self.force_redraw = True
                elif key == ord('e'): 
                    self.end_sync = self.global_frame
                    print(f"End: {self.end_sync}")
                    self.force_redraw = True
                
                # --- メイン操作 ---
                elif key == ord('r'): # Redo (Track)
                    self.is_playing = False
                    self.manual_targeting_part = None # ★重要: マニュアルモードを強制解除
                    
                    target_part = self.get_part_name_panel()
                    if target_part:
                        selected_cams = self.get_target_cameras_panel()
                        if selected_cams:
                            self.run_tracking_sequence([target_part], self.global_frame, target_cameras=selected_cams)
                
                elif key == ord('m'): # Manual Mode Toggle
                    self.is_playing = False
                    if self.manual_targeting_part:
                        # 既にモード中なら解除
                        self.manual_targeting_part = None
                        print("Manual Mode: OFF")
                    else:
                        target_part = self.get_part_name_panel()
                        if target_part:
                            self.manual_targeting_part = target_part
                            print(f"Manual Mode: ON [{target_part}]")
                    self.force_redraw = True
                
                elif key == ord('t'): # Toggle Status Panel
                    self.show_status_panel = not self.show_status_panel
                    self.force_redraw = True

                if self.is_playing:
                    if self.global_frame < self.total_frames - 1:
                        self.global_frame += 1
                        cv2.setTrackbarPos("Time", self.window_name, self.global_frame)
                    else:
                        self.is_playing = False

            for cap in self.caps.values(): cap.release()
            cv2.destroyAllWindows()
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Time", self.window_name, 0, self.total_frames, self.on_trackbar)
            cv2.setTrackbarPos("Time", self.window_name, self.global_frame)
            cv2.setMouseCallback(self.window_name, self.mouse_cb)
            
            while True:
                img = self.get_combined_view()
                cv2.imshow(self.window_name, img)
                
                delay = 1 if self.is_playing else 30
                key = cv2.waitKey(delay) & 0xFF
                
                if key == ord('w'): 
                    self.save_to_csv()
                    self.force_redraw = True
                elif key == 17: # Ctrl + Q
                    if self.confirm_quit(): break
                elif key == 32: # Space
                    self.is_playing = not self.is_playing
                elif key == 81 or key == 2424832: # Left
                    self.is_playing = False
                    if self.global_frame > 0:
                        self.global_frame -= 1
                        cv2.setTrackbarPos("Time", self.window_name, self.global_frame)
                        self.force_redraw = True
                elif key == 83 or key == 2555904: # Right
                    self.is_playing = False
                    if self.global_frame < self.total_frames:
                        self.global_frame += 1
                        cv2.setTrackbarPos("Time", self.window_name, self.global_frame)
                        self.force_redraw = True
                elif key == ord('s'): 
                    self.start_sync = self.global_frame
                    print(f"Start: {self.start_sync}")
                    self.force_redraw = True
                elif key == ord('e'): 
                    self.end_sync = self.global_frame
                    print(f"End: {self.end_sync}")
                    self.force_redraw = True
                
                # --- メイン操作 ---
                elif key == ord('r'): # Redo (Track)
                    self.is_playing = False
                    target_part = self.get_part_name_panel()
                    if target_part:
                        selected_cams = self.get_target_cameras_panel()
                        if selected_cams:
                            self.run_tracking_sequence([target_part], self.global_frame, target_cameras=selected_cams)
                
                elif key == ord('m'): # Manual Mode
                    self.is_playing = False
                    target_part = self.get_part_name_panel()
                    if target_part:
                        self.manual_targeting_part = target_part
                        self.force_redraw = True
                
                elif key == ord('t'): # Toggle Status Panel
                    self.show_status_panel = not self.show_status_panel
                    self.force_redraw = True

                if self.is_playing:
                    if self.global_frame < self.total_frames - 1:
                        self.global_frame += 1
                        cv2.setTrackbarPos("Time", self.window_name, self.global_frame)
                    else:
                        self.is_playing = False

            for cap in self.caps.values(): cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MultiCamTrackerApp()
    app.run()