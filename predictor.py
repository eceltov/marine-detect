from src.marine_detect.predict import predict_on_images, get_models, predict_on_image
import torch
import pickle
from pathFinder import get_image_filepaths

class DetectionBoxes:
   def __init__(self, boxes, counts):
      self.boxes = boxes
      self.counts = counts

def get_image_detection_boxes(models, filepath):
  # run the model
  res = predict_on_image(
    models = models,
    confs_threshold=[0.3, 0.4],
    image_input_path=filepath,
  )

  # extract detection boxes from the result
  boxes = []
  for r in res:
      for box in r.boxes.xyxy:
        boxes.append(box.view(1, box.shape[0]))
  if len(boxes) == 0:
     return torch.tensor([])
  
  boxes_tensor = torch.cat(boxes)
  return boxes_tensor.to("cpu")

def get_all_detection_boxes(filepaths):
  models = get_models(["models/FishInv.pt", "models/MegaFauna.pt"])
  detection_boxes = []
  detection_counts = []

  for filepath in filepaths:
    boxes = get_image_detection_boxes(models, filepath)
    
    detection_counts.append(boxes.shape[0])
    detection_boxes.append(boxes)

  return DetectionBoxes(detection_boxes, detection_counts)
    

filepaths = get_image_filepaths()
detection_boxes = get_all_detection_boxes(filepaths)

with open("detectionBoxes.pickle", 'wb') as handle:
  pickle.dump(detection_boxes, handle, protocol=pickle.HIGHEST_PROTOCOL)
