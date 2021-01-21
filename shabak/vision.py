import torch
import numpy as np
from torchvision import models
from PIL import Image
from torchvision import transforms
import json

model = models.resnet18(pretrained=True)
model.eval()
fc = model.fc
print(fc)

v = np.load('image_embedding.npy')
t = torch.from_numpy(v)
output = fc(t)

top10 = output.argsort()[-10:]
print('top10', top10)
top10_list = top10.tolist()
print('top10list', top10_list)

with open('imagenet_class_index.json') as classes:
    class_idx = json.load(classes)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    print([idx2label[idx] for idx in top10_list])

input_image = Image.open('rgbhint.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

with torch.no_grad():
    output = model(input_batch)

top10 = output[0].argsort()[-10:]
print(top10)
top10_list = top10.tolist()
print(top10_list)

with open('imagenet_class_index.json') as classes:
    class_idx = json.load(classes)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    print ([idx2label[idx] for idx in top10_list])
