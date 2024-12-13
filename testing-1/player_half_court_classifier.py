import json
import numpy as np
import cv2

# COURT_RoI = np.array([(345, 320), (935, 320), (1140, 720), (145, 720)], dtype=np.int32)


# for 640 * 640
COURT_RoI = np.array([
    (180, 285),  # 左上
    (480, 285), # 右上
    (580, 635), # 右下
    (90, 635)   # 左下
], dtype=np.int32)

# MIDLINE_Y = 475
MIDLINE_Y = 460


def classify_player(player_bbox):
    '''
    Args:
        player_bbox(list): list of bbox for player.
    Returns:
        str: Classification for the player ['Top Half', 'Bottom Half', 'Outside'].
    '''
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


if __name__ == '__main__':

    # show court range
    # img = cv2.imread('./test/images/1_13_14_mp4-0064_jpg.rf.2adf30f453f8268b054841769324b852.jpg')
    # cv2.polylines(img, [COURT_RoI], isClosed=True, color=(0, 255, 0), thickness=2)
    # cv2.imshow('Court ROI', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 取图片的宽度和高度
    # height, width, _ = img.shape
    # 输出图片的宽度和高度
    # print(f"Width: {width}, Height: {height}")

    # test classify_player
    # player_bbox_xyxy = [389.0, 384.0, 476.0, 654.0]
    # print(classify_player(player_bbox_xyxy))  # Bottom Half


    with open('results.json') as file:
        data = json.load(file)
        for img_name, detections in data.items():
            for detection in detections:
                print(img_name, ": ", classify_player(detection["bbox"]))
