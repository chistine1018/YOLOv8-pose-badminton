from ultralytics import YOLO

model = YOLO("best100.pt")
# source = 'test/images/06_mp4-0000_jpg.rf.61f83277d9de17e614072c758a18c7df.jpg'
# source = 'test/images'
# source = '1_13_14.mp4'
# source = '2_05_03.mp4'
source = '../Lab_3_Dataset-20241130T060726Z-001/Lab_3_Dataset/Fore_Back_Detection'
results = model.predict(source, imgsz=640, stream=True, save=True)
# 遍歷生成器並手動處理每個結果
for result in results:
    result.save()  # 儲存檢測結果





##############################################################################################################

# from ultralytics import YOLO
#
# # 載入模型
# model = YOLO("runs/detect/train4/weights/best20.pt")
#
# # 預測圖片來源，可以是本地路徑或是 URL
# # source = 'https://ultralytics.com/images/bus.jpg'
# source = 'test/images/06_mp4-0000_jpg.rf.61f83277d9de17e614072c758a18c7df.jpg'
#
# # 執行預測
# results = model.predict(source)
#
# # 取得預測結果中的第一個元素（如果有多個圖像）
# result = results[0]
#
# # 顯示結果
# result.show()  # 顯示圖片和檢測框
#
# # 保存結果（如果需要）
# result.save()  # 可選：保存結果到預設的 output 目錄存結果到預設的 output 目錄