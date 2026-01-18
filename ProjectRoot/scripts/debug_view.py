import cv2
import cv2.aruco as aruco
import sys

# === 設定 (d_b8.pyと同じにする) ===
ARUCO_DICT_ID = aruco.DICT_4X4_50
CHARUCO_SIZE = (6, 9)
SQUARE_LENGTH = 90.0
MARKER_LENGTH = 70.0
# =================================

def debug_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("画像が見つかりません")
        return

    # 辞書とボード
    try:
        aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    except AttributeError:
        aruco_dict = aruco.Dictionary_get(ARUCO_DICT_ID)

    # パラメータ設定
    params = aruco.DetectorParameters() if hasattr(aruco, 'DetectorParameters') else aruco.DetectorParameters_create()
    
    # =========================================================
    #  【重要】対・自然物（葉っぱ）用の厳格フィルタ設定
    # =========================================================

    # 1. 形の整い具合 (デフォルト 0.05 -> 0.01)
    # 葉っぱのような「ギザギザした四角」や「歪んだ四角」を許さず、
    # 直線で構成された綺麗な四角形だけを候補にします。
    params.polygonalApproxAccuracyRate = 0.03

    # 2. サイズ制限 (小さすぎるゴミを無視)
    # 画面の2%以下のサイズはノイズとして無視します。
    # ※遠くのボードが映らなくなる場合は 0.015 くらいまで下げてください。
    params.minMarkerPerimeterRate = 0.005

    # 3. ノイズ除去（ウィンドウサイズ拡大）
    # 葉っぱの葉脈のような細かい模様を無視するために、
    # 2値化のウィンドウサイズ（Min）を大きくします（デフォルト3 -> 15）。
    params.adaptiveThreshWinSizeMin = 15
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 10

    # 4. 近接ノイズの排除
    # 葉っぱは密集しているので、近すぎるコーナー候補は除外します。
    params.minCornerDistanceRate = 0.05

    # 5. マーカーの「黒枠」チェックを厳しくする
    # 葉っぱには明確な黒枠がないため、ここで弾きます。
    params.maxErroneousBitsInBorderRate = 0.15

    # =========================================================
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # マーカー検出
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=params)

    # 描画
    debug_img = img.copy()
    
    # 1. 棄却された候補（四角形っぽいがマーカーではないと判断されたもの）を赤で描画
    # 設定がうまくいっていれば、葉っぱ部分に赤枠が出なくなる（無視される）はずです。
    if len(rejected) > 0:
        aruco.drawDetectedMarkers(debug_img, rejected, borderColor=(0, 0, 255))
    
    # 2. 正しく検出されたマーカーを緑で描画
    if len(corners) > 0:
        aruco.drawDetectedMarkers(debug_img, corners, ids, borderColor=(0, 255, 0))
        print(f"✅ 検出成功: {len(corners)}個")
    else:
        print("❌ 検出数 0個")

    # 保存
    cv2.imwrite("debug_result_strict.jpg", debug_img)
    print("結果を 'debug_result_strict.jpg' に保存しました。")
    print("確認ポイント:")
    print("・緑色の枠がボード上のマーカーに出ているか？")
    print("・赤色の枠（ノイズ候補）が、葉っぱや芝生から消えているか（少なくなっているか）？")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_detection(sys.argv[1])
    else:
        # 画像パスを直接指定する場合はここを書き換えてください
        # debug_detection("../data/pixel/pixel_001.jpg")
        print("Usage: python debug_view.py [画像パス]")