![image](https://github.com/user-attachments/assets/eb98e480-b8f2-4cb5-9f8c-6741de3bd371)


<h1>協助注意以下事項：  </h1> 

1. 有關Label資料集在Roboflow上的幀數設定，請同學設定為10 fps/sec，可以減少需標記圖片的數量
2. 標記的操作這兩天會再錄影片給同學參考，再請同學注意一下公告的更新  
3. 標記結果的繳交同於lab2，請在12/9(一)23:59前上傳.zip檔到E3上  
4. 正反手的判斷可以單純檢查球員慣用手於身體左側或右側即可(測資的球員都是右撇子)

可參考 Lab3_Overview.pdf, Lab 3.YOLO-pose.pdf


<h2>yolo_dataset_download.py   </h2>

1. 下載羽球員骨架標記dataset
2. 一個是下載640 * 640 的dataset
3. 一個是下載1920 * 1080 的dataset

  
<h2>testing-1/yolo_pose.py   </h2>

1. 基於原本yolo-pose模型進行訓練
2. 這邊epochs 我是設定20 (100的時候貌似overfitting測試效果不大好)

  

<h2>testing-1/yolo_predict.py   </h2>

1. 基於訓練好的模型去預測結果

  

<h2>testing-1/yolo_pose_predict_and_save_keypoints.py   </h2>

1. 將keypoints存成json格式


```json
{
  "image1.jpg": [
    {
      "bbox": [100.0, 200.0, 150.0, 300.0],
      "keypoints": [
        [120.0, 210.0],
        [125.0, 215.0],
        ...
      ]
    },
    {
      "bbox": [50.0, 180.0, 90.0, 250.0],
      "keypoints": [
        [60.0, 190.0],
        [65.0, 195.0],
        ...
      ]
    }
  ],
  "image2.jpg": [
    ...
  ]
}
```
  

<h2>testing-1/player_half_court_classifier.py   </h2>

1. 將keypoints json讀出來
2. 根據bbox 判斷人物在上半場還是下半場
3. 要小心圖片解析度有可能影響球場ROI

```

# 這是FOR 1920 * 1080的球場
# COURT_RoI = np.array([(345, 320), (935, 320), (1140, 720), (145, 720)], dtype=np.int32)
# MIDLINE_Y = 475


# 這是FOR 640 * 640的球場
COURT_RoI = np.array([
    (180, 285),  # 左上
    (480, 285), # 右上
    (580, 635), # 右下
    (90, 635)   # 左下
], dtype=np.int32)

MIDLINE_Y = 460
```

```
    # show court range 畫出球場roi範圍
    img = cv2.imread('./test/images/1_13_14_mp4-0064_jpg.rf.2adf30f453f8268b054841769324b852.jpg')
    cv2.polylines(img, [COURT_RoI], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow('Court ROI', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

  
<h2>testing-1/anson_lab3_predict_pose.py   </h2>

1. 先判斷球員位於上半場還下半場
2. 在判斷手腕與鼻子x軸的大小去決定正反手

  
<h2>testing-1/kenny_lab3_predict_pose.py   </h2>

1. KENNY哥算法

  
