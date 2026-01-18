はい、先ほどのフォーマットに合わせつつ、**システム構成図（Architecture）** と **`uv` による環境構築** を反映したバージョンです。

これをコピーして `README.md` に貼り付けてください。

---

```markdown
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

```

### 2. Install Dependencies

依存ライブラリを一括インストールします。

```bash
uv pip install -r requirements.txt

```

## Usage Workflow

図のフローに従って順番に実行します。

### Phase 1: Preparation

動画の同期と画像切り出しを行います。

```bash
python sync_manager.py

```

### Phase 2: Calibration

カメラの位置関係（外部パラメータ）を計算します。

```bash
python calibration.py

```

### Phase 3: 2D Tracking

動画上で対象人物の関節をトラッキングし、2D座標を取得します。

```bash
python tracker.py

```

### Phase 4: 3D Reconstruction

2D座標とカメラパラメータから3D座標を復元し、結果を描画します。

```bash
python reconstruct.py

```

## File Structure

```text
.
├── sync_manager.py    # 動画同期・前処理
├── calibration.py     # カメラパラメータ推定
├── tracker.py         # 2Dトラッキングツール
├── reconstruct.py     # 3D復元・可視化
├── architecture.png   # システム構成図
├── requirements.txt   # 依存ライブラリ
├── data/              # 動画・CSVデータ保存先
├── calibration_data/  # キャリブレーション用画像置き場
├── system_matrices/   # 計算されたパラメータ出力先
└── README.md          # ドキュメント

```

```

```
