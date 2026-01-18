import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
import threading
import subprocess
import json
import re  # 正規表現用に追加

# --- ★設定エリア★ ---
# ユーザーの環境に合わせてパスを設定
DEFAULT_VIDEO_DIR = '/home/shinichirou/Desktop/douki/21回目'
OUTPUT_BASE_DIR = "../data"
SYNC_CONFIG_FILE = "../data/sync_config.json"

CAMERA_CONFIG = [
    ("pixel", "pixel.mp4"),
    ("oppo",  "oppo.mp4"),
    ("oppo1", "oppo1.mp4"),
    ("oppo2", "oppo2.mp4"),
]
# --------------------

class MultiCamSyncApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Camera Sync (Frame Sync Edition)")
        
        try:
            self.root.attributes('-zoomed', True)
        except Exception:
            self.root.geometry("1600x1000")

        self.num_cams = len(CAMERA_CONFIG)
        self.caps = [None] * self.num_cams
        self.paths = [None] * self.num_cams
        self.offsets = [0] * self.num_cams
        self.total_frames = [0] * self.num_cams
        self.fps_list = [30.0] * self.num_cams
        self.is_playing = [False] * self.num_cams 
        
        # 同期情報保持用
        self.sync_info = [""] * self.num_cams
        
        self.panels = [] 
        self.labels_info = []
        self.labels_title = []
        self.labels_ts = [] 
        self.sliders = []     
        self.btns_play = []   
        
        self._setup_ui()
        # 起動直後に自動読み込み
        self.root.after(100, self.auto_load_all_videos)

    def _setup_ui(self):
        # 上部に一括操作エリア
        frame_top = tk.Frame(self.root, bg="#222", pady=5)
        frame_top.pack(fill="x", side="top")
        
        tk.Button(frame_top, text="📂 フォルダ選択して再読込", font=("Arial", 12), 
                  bg="#555", fg="white", command=self.change_directory).pack(side="left", padx=20)
        
        tk.Label(frame_top, text="※ログ(Frame=XX)に基づいて開始位置を自動設定しています", 
                 bg="#222", fg="#aaa").pack(side="left", padx=10)

        frame_video = tk.Frame(self.root, bg="#111")
        frame_video.pack(expand=True, fill="both")

        for i, (cam_name, default_file) in enumerate(CAMERA_CONFIG):
            row = i // 2
            col = i % 2
            
            p_frame = tk.Frame(frame_video, bg="#333", bd=1, relief="solid")
            p_frame.grid(row=row, column=col, sticky="nsew", padx=1, pady=1)
            
            # ヘッダー
            header_frame = tk.Frame(p_frame, bg="#222")
            header_frame.pack(fill="x")
            
            lbl_title = tk.Label(header_frame, text=f"Camera: {cam_name}", bg="#222", fg="#eee", font=("Arial", 12, "bold"))
            lbl_title.pack(side="left", padx=5)
            
            # 同期情報表示用ラベル
            lbl_ts = tk.Label(header_frame, text="Sync: None", bg="#222", fg="#aaa", font=("Arial", 10))
            lbl_ts.pack(side="right", padx=5)
            
            # 映像表示エリア
            lbl_img = tk.Label(p_frame, text="No Video", bg="black", fg="#555")
            lbl_img.pack(expand=True, fill="both")

            ctrl_frame = tk.Frame(p_frame, bg="#444")
            ctrl_frame.pack(fill="x")

            slider = tk.Scale(ctrl_frame, from_=0, to=100, orient="horizontal", showvalue=False,
                              bg="#444", fg="white", troughcolor="#666", activebackground="orange",
                              command=lambda val, idx=i: self.on_slider_change(idx, val))
            slider.pack(fill="x", padx=5, pady=2)
            
            btn_row = tk.Frame(ctrl_frame, bg="#444")
            btn_row.pack(fill="x", pady=4)

            tk.Button(btn_row, text="Open", width=5, bg="#666", fg="white", 
                      command=lambda idx=i: self.load_video_dialog(idx)).pack(side="left", padx=4)

            tk.Button(btn_row, text="<<", width=4, command=lambda idx=i: self.step_frame(idx, -50)).pack(side="left")
            tk.Button(btn_row, text="<", width=4, command=lambda idx=i: self.step_frame(idx, -1)).pack(side="left")
            btn_play = tk.Button(btn_row, text="▶", width=5, bg="#cfc", command=lambda idx=i: self.toggle_play(idx))
            btn_play.pack(side="left", padx=8)
            tk.Button(btn_row, text=">", width=4, command=lambda idx=i: self.step_frame(idx, 1)).pack(side="left")
            tk.Button(btn_row, text=">>", width=4, command=lambda idx=i: self.step_frame(idx, 50)).pack(side="left")
            
            lbl_info = tk.Label(btn_row, text="Frame: 0", font=("Arial", 12), bg="#444", fg="white")
            lbl_info.pack(side="right", padx=10)
            
            self.panels.append(lbl_img)
            self.labels_info.append(lbl_info)
            self.labels_title.append(lbl_title)
            self.labels_ts.append(lbl_ts)
            self.sliders.append(slider)
            self.btns_play.append(btn_play)

        frame_video.grid_columnconfigure(0, weight=1)
        frame_video.grid_columnconfigure(1, weight=1)
        frame_video.grid_rowconfigure(0, weight=1)
        frame_video.grid_rowconfigure(1, weight=1)

        frame_bottom = tk.Frame(self.root, height=80, bg="#222")
        frame_bottom.pack(fill="x", side="bottom")

        btn_export = tk.Button(frame_bottom, text="★ SAVE SYNC & EXPORT (FFmpeg) ★", 
                               font=("Arial", 18, "bold"), bg="orange", fg="black",
                               command=self.start_export_thread)
        btn_export.pack(pady=20)

    # --- ディレクトリ変更 ---
    def change_directory(self):
        global DEFAULT_VIDEO_DIR
        new_dir = filedialog.askdirectory(initialdir=DEFAULT_VIDEO_DIR)
        if new_dir:
            DEFAULT_VIDEO_DIR = new_dir
            self.auto_load_all_videos()

    # --- 自動読み込みロジック ---
    def auto_load_all_videos(self):
        print(f"Loading videos from: {DEFAULT_VIDEO_DIR}")
        for i, (cam_name, filename) in enumerate(CAMERA_CONFIG):
            path = os.path.join(DEFAULT_VIDEO_DIR, filename)
            if os.path.exists(path):
                self.load_video_from_path(i, path)
            else:
                self.labels_title[i].config(text=f"Camera: {cam_name} (Not Found)", bg="#440000")

    def load_video_from_path(self, idx, path):
        # 既存リソース解放
        if self.caps[idx] is not None:
            self.caps[idx].release()

        self.paths[idx] = path
        self.caps[idx] = cv2.VideoCapture(path)
        
        if not self.caps[idx].isOpened():
            print(f"Error opening video: {path}")
            return

        total = int(self.caps[idx].get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.caps[idx].get(cv2.CAP_PROP_FPS)
        self.total_frames[idx] = total
        self.fps_list[idx] = fps
        
        # デフォルトは0スタート
        self.offsets[idx] = 0
        self.sliders[idx].config(to=total)
        
        cam_name = CAMERA_CONFIG[idx][0]
        filename = os.path.basename(path)
        self.labels_title[idx].config(text=f"Camera: {cam_name} ({filename}) - {fps:.2f}fps", bg="#005500")

        # --- ログファイル読み込み & 自動同期 ---
        # 動画と同じ名前の .txt を探す (例: pixel.mp4 -> pixel.txt)
        txt_path = os.path.splitext(path)[0] + ".txt"
        
        sync_msg = "Manual"
        found_frame = None

        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # 正規表現で "Frame=数字" を探す
                    match = re.search(r"Frame=(\d+)", content)
                    if match:
                        found_frame = int(match.group(1))
                        # ★ここでオフセット（開始位置）を自動設定！
                        self.offsets[idx] = found_frame
                        sync_msg = f"Auto Log: Frame={found_frame}"
                        print(f"[{cam_name}] Log Found: Start Frame = {found_frame}")
                    else:
                        sync_msg = "Log Found (No Frame data)"
            except Exception as e:
                print(f"Log read error: {e}")
                sync_msg = "Log Error"
        else:
            sync_msg = "No Log File"
        
        self.sync_info[idx] = sync_msg
        self.labels_ts[idx].config(text=sync_msg, fg="#00ff00" if found_frame else "#aaa")
        
        # スライダーとプレビューを同期位置に合わせる
        self.sliders[idx].set(self.offsets[idx])
        self.precise_seek(idx, self.offsets[idx])
        self.update_display(idx, seek_done=True)

    def load_video_dialog(self, idx):
        path = filedialog.askopenfilename(initialdir=DEFAULT_VIDEO_DIR, 
                                          title=f"Select video for {CAMERA_CONFIG[idx][0]}",
                                          filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.load_video_from_path(idx, path)

    # --- 再生制御 ---
    def toggle_play(self, idx):
        if self.caps[idx] is None: return
        if self.is_playing[idx]:
            self.is_playing[idx] = False
            self.btns_play[idx].config(text="▶", bg="#cfc")
        else:
            self.is_playing[idx] = True
            self.btns_play[idx].config(text="||", bg="#fcc")
            self.play_loop(idx)

    def play_loop(self, idx):
        if not self.is_playing[idx]: return
        if self.offsets[idx] < self.total_frames[idx] - 1:
            self.offsets[idx] += 1
            self.update_display(idx)
            self.sliders[idx].set(self.offsets[idx])
            self.root.after(33, lambda: self.play_loop(idx))
        else:
            self.is_playing[idx] = False
            self.btns_play[idx].config(text="▶", bg="#cfc")

    def on_slider_change(self, idx, val):
        if self.caps[idx] is None: return
        frame_num = int(val)
        if frame_num != self.offsets[idx]: 
            self.offsets[idx] = frame_num
            self.precise_seek(idx, frame_num)
            self.update_display(idx, seek_done=True)

    def step_frame(self, idx, step):
        if self.caps[idx] is None: return
        if self.is_playing[idx]: self.toggle_play(idx)
        new_frame = self.offsets[idx] + step
        new_frame = max(0, min(new_frame, self.total_frames[idx] - 1))
        self.offsets[idx] = new_frame
        self.precise_seek(idx, new_frame)
        self.sliders[idx].set(new_frame)
        self.update_display(idx, seek_done=True)

    def precise_seek(self, idx, target_frame):
        cap = self.caps[idx]
        if cap is None: return
        # 高速シークのために直前に飛んでから微調整
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current == target_frame: return
        
        if target_frame > 50 and abs(current - target_frame) > 50:
             cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # 確実に合わせるため、セット後に確認してズレてたら読み捨てる
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    def update_display(self, idx, seek_done=False):
        if self.caps[idx] is None: return
        cap = self.caps[idx]
        
        if not seek_done:
             cap.set(cv2.CAP_PROP_POS_FRAMES, self.offsets[idx])
             
        ret, frame = cap.read()
        if ret:
            # プレビュー用にリサイズ (処理落ち防止)
            display_h = 880 
            h, w = frame.shape[:2]
            scale = min(display_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # OpenCV(BGR) -> PIL(RGB)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.panels[idx].configure(image=imgtk, text="")
            self.panels[idx].image = imgtk
            
            # ラベル更新
            status = "Manual"
            if "Auto" in self.sync_info[idx]: status = "Auto"
            self.labels_info[idx].configure(text=f"Start: {self.offsets[idx]} ({status})")
        else:
            # 読み込み失敗時（末尾など）
            self.panels[idx].configure(image='', text="End of Video")

    # --- 書き出し処理 ---
    def start_export_thread(self):
        loaded_count = sum(1 for c in self.caps if c is not None)
        if loaded_count == 0:
            messagebox.showwarning("警告", "動画がありません。")
            return
        if not messagebox.askyesno("確認", f"{loaded_count}台の書き出しを開始しますか？\n(指定した開始フレーム以降を書き出します)"):
            return
        thread = threading.Thread(target=self.run_export_parallel)
        thread.start()

    def run_export_parallel(self):
        print("--- 設定保存開始 ---")
        sync_data = {}
        for i, (cam_name, _) in enumerate(CAMERA_CONFIG):
            if self.caps[i] is None: continue
            sync_data[cam_name] = {
                "offset_frame": self.offsets[i],
                "video_path": self.paths[i]
            }
        try:
            os.makedirs(os.path.dirname(SYNC_CONFIG_FILE), exist_ok=True)
            with open(SYNC_CONFIG_FILE, 'w') as f:
                json.dump(sync_data, f, indent=4)
        except Exception as e:
            print(f"エラー: {e}")
            return

        print("--- FFmpeg 書き出し開始 ---")
        processes = []
        for i, (cam_name, _) in enumerate(CAMERA_CONFIG):
            if self.caps[i] is None: continue
            video_path = self.paths[i]
            save_dir = os.path.join(OUTPUT_BASE_DIR, cam_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # 設定したオフセット（開始フレーム）
            start_frame = self.offsets[i]
            
            # 連番画像として出力 (動画にする場合は .mp4 に変えてください)
            output_pattern = os.path.join(save_dir, f"{cam_name}_%04d.jpg")
            
            # FFmpegコマンド: select=gte(n, start_frame) でトリミング
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f"select=gte(n\,{start_frame})",
                '-vsync', '0',
                '-q:v', '2',
                '-start_number', '0',
                output_pattern
            ]
            
            # 非同期実行
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append((cam_name, p))

        # 完了待機
        for name, p in processes:
            p.wait()
            print(f"Done: {name}")
            
        messagebox.showinfo("完了", "書き出し完了！")

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiCamSyncApp(root)
    root.mainloop()