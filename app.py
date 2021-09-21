from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'

class_dict = {0: 'BRVO', 1: 'Normal'}

model = models.resnet50(pretrained=False)
first_conv_layer=[nn.Conv2d(1,3, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)] #adapter for 1 ch input
first_conv_layer.extend([(model.conv1)])
model.conv1=nn.Sequential(*first_conv_layer)
classifier = nn.Linear(2048, 2)
model.fc = classifier
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)

checkpoint = torch.load('model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state'], strict=False)

train2_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

def predict_label(img_path):
    image = Image.open(Path(img_path))
    input = train2_transforms(image)
    input = input[1,:,:].unsqueeze(0).unsqueeze(0)
    output = model(input)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    return class_dict[prediction]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)