from ultralytics import YOLO
# Load a model
model = YOLO("yolov8n-pose.pt")
# load a pretrained model (recommended for training) # Train the model
results = model.train(data="data.yaml", epochs=100,
                      # batch=64,
                      imgsz=640,
                      device=0,
                      workers=0)