import json
import numpy as np
import cv2

COURT_RoI = np.array([(345, 320), (935, 320), (1140, 720), (145, 720)], dtype=np.int32)

# for 640 * 640
# COURT_RoI = np.array([
#     (180, 285),  # 左上
#     (480, 285), # 右上
#     (580, 635), # 右下
#     (90, 635)   # 左下
# ], dtype=np.int32)

MIDLINE_Y = 475


# MIDLINE_Y = 460


def classify_player(player_bbox):
    """
    Args:
        player_bbox (list): List of bbox for player in [x1, y1, x2, y2] format.
    Returns:
        str: Classification for the player ['Top Half', 'Bottom Half', 'Outside Court'].
    """
    x1, y1, x2, y2 = player_bbox
    center_bottom = ((x1 + x2) // 2, y2)

    is_inside = cv2.pointPolygonTest(COURT_RoI, center_bottom, False) >= 0

    if not is_inside:
        return "Outside Court"
    else:
        if center_bottom[1] < MIDLINE_Y:
            return "Top Half"
        else:
            return "Bottom Half"


NOSE = 0
LEFT_WRIST = 9
RIGHT_WRIST = 10


def classify_action(keypoints, is_bottom_half):
    """
    根據手腕相對於鼻子的水平位置，判斷是 forehand 或 backhand。

    Args:
        keypoints (list): 含有關鍵點 [x, y] 的列表。

    Returns:
        str: 動作分類結果，forehand 或 backhand，若無法判斷則返回 unknown。
    """
    try:
        # 提取鼻子和手腕的 x 座標
        nose_x = keypoints[NOSE][0]
        left_wrist_x = keypoints[LEFT_WRIST][0]
        right_wrist_x = keypoints[RIGHT_WRIST][0]

        # 印出關鍵點座標
        # print(f"Nose X: {nose_x}, Right Wrist X: {right_wrist_x}")

        # 判斷手腕相對於鼻子的水平位置
        if (is_bottom_half):
            if right_wrist_x > nose_x:  # 右手腕在鼻子右邊
                return "forehand"
            elif right_wrist_x < nose_x:  # 左手腕在鼻子右邊
                return "backhand"
            else:
                return "unknown"
        else:
            if right_wrist_x < nose_x:  # 右手腕在鼻子右邊
                return "forehand"
            elif right_wrist_x > nose_x:  # 左手腕在鼻子右邊
                return "backhand"
            else:
                return "unknown"

    except (IndexError, TypeError):
        # 當關鍵點不足或格式不正確時，返回 unknown
        print("Error: Missing or invalid keypoints")
        return "unknown"


if __name__ == '__main__':
    # 读取 JSON 文件
    with open('resultstest.json') as file:
        data = json.load(file)

        # 遍历每张图片的检测结果
        for img_name, detections in data.items():
            for detection in detections:
                bbox = detection["bbox"]
                keypoints = detection["keypoints"]

                # 分类场地位置
                position = classify_player(bbox)

                # 分类动作类型
                action = classify_action(keypoints, position == "Bottom Half")

                # 打印结果
                print(f"{img_name}: {position} {action}")
