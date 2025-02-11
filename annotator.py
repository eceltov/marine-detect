from src.marine_detect.predict import predict_on_images, get_models, predict_on_image
import sys
import torch
import pickle
from pathFinder import get_image_filepaths

filepaths = get_image_filepaths()
models = get_models(["models/FishInv.pt", "models/MegaFauna.pt"])

for filepath in filepaths:
  out_filepath = filepath.replace("MVK", "MVKout")
  res = predict_on_image(
      models = models,
      confs_threshold=[0.3, 0.4],
      image_input_path=filepath,
      image_output_path=out_filepath,
  )
