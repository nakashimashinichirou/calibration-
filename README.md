# Multi-View 3D Motion Capture System

市販のカメラ（スマホ等）を複数台用いて、動画の同期から3D骨格復元までを行う自家製モーションキャプチャシステムです。
PythonとOpenCVを使用し、キャリブレーション・2D追跡・3D再構成（DLT法）・スムージング処理を一貫して行います。

## System Architecture

データフローと処理パイプライン：
![System Architecture](./architecture.png)

## Features

* **Synchronization (`sync_manager.py`)**: 複数カメラの動画開始位置をフレーム単位で同期し、連番画像を生成。
* **Calibration (`calibration.py`)**: ChArUcoボードを用いてカメラの内部パラメータ・外部パラメータ（位置関係）を推定。
* **2D Tracking (`tracker.py`)**: 同期された映像上で関節位置を追跡（CSRT Tracker + 手動修正UI）。
* **3D Reconstruction (`reconstruct.py`)**: 
    * 多視点三角測量による3D座標計算。
    * OneEuroFilterによる時系列データのスムージング（ジッター除去）。
    * 3D骨格アニメーションの可視化。

## Environment (uv Recommended)

パッケージ管理には `uv` を使用しています。以下の手順で高速に環境を構築可能です。

### 1. Setup Virtual Environment
`uv` をインストールし、仮想環境を作成・有効化します。

```bash
# uvのインストール (未インストールの場合)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# 仮想環境の作成と有効化
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
