# •	INSTALL(02/06) Figure out stupid install problems: maybe the following worked???
# activate the triangle_transformers virtual env
#o	conda install pydantic
#•	conda install pytorch torchvision torchaudio cpuonly -c pytorch
#o	conda install -c conda-forge transformers
#o	conda install -c conda-forge matplotlib
#o	conda install -c conda-forge accelerate
#   pip install phonemizer

# figure out how to use Meta's DINO v3

import os
import torch

import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor
from transformers import AutoModel
from transformers.image_utils import load_image

# image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
plt.imshow(image)
plt.show()

model_name = "facebook/dinov2-giant"

#pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name, 
    device_map="auto", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)
