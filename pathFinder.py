import os

def get_image_filepaths():
  dataset_path = "/mnt/c/school/videa/MVK"
  filepaths = []

  for dirname in sorted(os.listdir(dataset_path)):
    dirpath = os.path.join(dataset_path, dirname)
    for fn in sorted(os.listdir(dirpath)):
      filename = os.path.join(dirpath, fn)
      filepaths.append(filename)

  return filepaths
