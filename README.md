

---



スマホカメラを複数台用いて、動画の同期から3D骨格復元までを行う自家製モーションキャプチャシステムです。




## Features

* **データ作成(`sync_manager.py`)**: 複数カメラの動画開始位置をフレーム単位で同期し、連番画像を生成。
* **カメラ間のキャリブレーション (`calibration.py`)**: ChArUcoボードを用いてカメラの位置関係を推定。
* **関節のトラッキング (`tracker.py`)**: 同期された映像上で関節位置を追跡。
* **3D化 (`animation.py`)**: 
    * 3D骨格アニメーションの可視化。

## Environment (uv Recommended)

パッケージ管理には `uv` を使用しています。


## File Structure

```text
.
├── sync_manager.py    # 動画同期・前処理
├── calibration.py     # カメラパラメータ推定
├── tracker.py         # 2Dトラッキングツール
├── animation.py     # 3D復元・可視化
├── requirements.txt   # 依存ライブラリ
├── data/              # 動画・CSVデータ保存先
├── calibration_data/  # キャリブレーション用画像置き場
├── system_matrices/   # 計算されたパラメータ出力先
└── README.md          # ドキュメント

```

```

```
