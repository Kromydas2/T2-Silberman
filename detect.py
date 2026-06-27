from ultralytics import YOLO
import cv2

# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolo26n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("yolo26n_ncnn_model")

# Run inference
results = ncnn_model("https://ultralytics.com/images/bus.jpg")
speed_dict = results[0].speed
res = results[0]

total_time_ms = sum(speed_dict.values())
fps = 1000 / total_time_ms

print(f"Total Inference Time: {total_time_ms:.2f} ms")
print(f"Frames Per Second (FPS): {fps:.2f}")

cv2.imshow("YOLO26 Detection", res.plot())
cv2.waitKey(0)
cv2.destroyAllWindows()