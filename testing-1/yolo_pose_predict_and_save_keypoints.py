import json
from ultralytics import YOLO

model = YOLO("best20.pt")
# source = 'test/images/06_mp4-0000_jpg.rf.61f83277d9de17e614072c758a18c7df.jpg'
# source = 'test/images'
# source = '1_13_14.mp4'
# YOLO 允許直接指定一個目錄，直接對目錄下的所有影像進行預測
results = model.predict(source="../Lab_3_Dataset-20241130T060726Z-001/Lab_3_Dataset/Fore_Back_Detection")
results_data = {}
for result in results:
    # 取出預測結果中儲存的影像的路徑，並僅取當中的檔名存為 image_name
    image_name = result.path.split('/')[-1]
    image_data = []

    # 逐一取出每張圖檢測結果的每個檢測資訊(每組骨架17組座標及bbox左上、右下角點座標-xyxy)
    for item in result:
        image_data.append({
            "bbox": item.boxes.xyxy[0].tolist(),
            "keypoints": item.keypoints.xy[0].tolist()
        })
    results_data[image_name] = image_data
with open("resultstest.json", "w") as f:
    json.dump(results_data, f, indent=4)
