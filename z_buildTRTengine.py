from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-pose.pt")  # load a pretrained model (recommended for training)
success = model.export(format="engine", device=[0, 1])  # export the model to engine format
assert success
