
from ultralytics import YOLO
import cv2
import os
import numpy as np


def calculate_angle(p1, p2, p3):
    """
    計算三個點之間的夾角 (p1, p2, p3 表示關鍵點坐標 [x, y])
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


def classify_hand_and_foot_action(keypoints, facing_camera):
    """
    基於17個關鍵點和視角來判斷動作類型（正手或反手），並結合腳步檢測。
    :param keypoints: 人物的關鍵點數據 (shape: [17, 3])
    :param facing_camera: 布爾值，True 表示面向鏡頭，False 表示背對鏡頭
    :return: 動作類型 ("forehand" 或 "backhand") 和腳步描述
    """
    # 關鍵點索引
    nose_idx = 0
    right_wrist_idx = 10
    left_shoulder_idx = 6
    right_shoulder_idx = 7
    right_elbow_idx = 8
    right_ankle_idx = 16
    left_ankle_idx = 15

    # 提取關鍵點座標
    nose = keypoints[nose_idx][:2]
    right_wrist = keypoints[right_wrist_idx][:2]
    left_shoulder = keypoints[left_shoulder_idx][:2]
    right_shoulder = keypoints[right_shoulder_idx][:2]
    right_elbow = keypoints[right_elbow_idx][:2]
    right_ankle = keypoints[right_ankle_idx][:2]
    left_ankle = keypoints[left_ankle_idx][:2]

    # 手部動作判斷
    if facing_camera:
        # 如果面向鏡頭
        is_backhand = right_wrist[0] > nose[0] and right_wrist[0] > right_shoulder[0]
    else:
        # 如果背對鏡頭
        is_backhand = right_wrist[0] < nose[0] and right_wrist[0] < left_shoulder[0]

    # 計算右手腕、右肘、右肩的夾角
    angle = calculate_angle(right_wrist, right_elbow, right_shoulder)

    # 判斷腳步動作
    foot_action = "stable"
    if abs(right_ankle[0] - left_ankle[0]) > 50:  # 假設 50 是跨步閾值
        foot_action = "stepping"
    elif right_ankle[0] > left_ankle[0]:
        foot_action = "right foot dominant"
    elif left_ankle[0] > right_ankle[0]:
        foot_action = "left foot dominant"

    # 綜合判斷
    hand_action = "backhand" if is_backhand and angle > 90 else "forehand"

    return hand_action, foot_action




if __name__ == "__main__":
    # 載入訓練好的模型
    model = YOLO('best20.pt')  # 訓練過程中保存的最佳模型

    # 設定影片路徑和輸出影片檔案
    input_video = '1_13_14.mp4'
    output_video = './test/outputs/annotated_video.mp4'

    # 開啟影片
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"無法打開影片: {input_video}")
        exit()

    # 取得影片的基本資訊
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 設定影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 編碼
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 讀取完所有幀

        # 對每一幀進行推論
        results_pred = model.predict(
            source=frame,  # 傳遞當前幀
            save=False,  # 不直接保存模型內建的標註結果，改用自訂繪製
            device=0,  # 使用 GPU
            imgsz=640  # 圖片大小
        )

        # 遍歷每個人
        for result in results_pred:
            for keypoints, box in zip(result.keypoints, result.boxes.xyxy):
                # 從 Keypoints 對象中提取數據並轉為 NumPy
                keypoints = keypoints.data.cpu().numpy()[0]  # 提取數據，轉為 (17, 3) 格式
                box = box.cpu().numpy()  # 同樣處理 Bounding Box

                # 根據人物在畫面中的位置判斷視角
                person_center_y = (box[1] + box[3]) / 2
                image_height = frame.shape[0]

                # 視角判斷
                facing_camera = person_center_y < image_height / 2  # 上方人物認為面向鏡頭

                # 判斷動作類型（Forehand 或 Backhand）和腳步動作
                hand_action, foot_action = classify_hand_and_foot_action(keypoints, facing_camera)

                # 繪製標註
                color = (0, 255, 0) if hand_action == "forehand" else (0, 0, 255)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)  # 繪製框
                cv2.putText(
                    frame, f"{hand_action} | {foot_action}", (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

        # 將標註後的幀寫入新影片
        out.write(frame)

    # 釋放資源
    cap.release()
    out.release()

    print(f"已將標註影片保存為：{output_video}")