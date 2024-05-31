import matplotlib.pyplot as plt
from lime import lime_image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from matplotlib.pyplot import imshow
from io import BytesIO
import base64
import random
import tensorflow as tf
import warnings
from skimage import img_as_ubyte
from torchvision import models
from torch import nn
import sys
import time
from torch.autograd import Variable
import face_recognition
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch
from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename
from skimage.segmentation import mark_boundaries
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

# Interaction with the OS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Used for DL applications, computer vision related processes

# For image preprocessing

# Combines dataset & sampler to provide iterable over the dataset


# To recognise face from extracted frames

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation


# 'nn' Help us in creating & training of neural network

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""

detectOutput = []

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

deepfake_model = tf.keras.models.load_model('deepfake-detection-model.h5')
print("model1 loaded")

# Creating Model Architecture


class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        # returns a model pretrained on ImageNet dataset
        model = models.resnext50_32x4d(pretrained=True)

        # Sequential allows us to compose modules nn together
        self.model = nn.Sequential(*list(model.children())[:-2])

        # RNN to an input sequence
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

        # Activation function
        self.relu = nn.LeakyReLU()

        # Dropping out units (hidden & visible) from NN, to avoid overfitting
        self.dp = nn.Dropout(0.4)

        # A module that creates single layer feed forward network with n inputs and m outputs
        self.linear1 = nn.Linear(2048, num_classes)

        # Applies 2D average adaptive pooling over an input signal composed of several input planes
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape

        # new view of array with same data
        x = x.view(batch_size*seq_length, c, h, w)

        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


model = Model(2)
path_to_model = 'model_93_acc_100_frames_celeb_FF_data.pt'
model.load_state_dict(torch.load(
    path_to_model, map_location=torch.device('cpu')))
model.eval()
print("model2 loaded")

im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(
    mean=-1*np.divide(mean, std), std=np.divide([1, 1, 1], std))

# For image manipulation


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png', image*255)
    return image

# For prediction of output


def predict(model, img, path='./'):
    # use this command for gpu
    # fmap, logits = model(img.to('cuda'))
    fmap, logits = model(img.to())
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item()*100
    print('confidence of prediction: ',
          logits[:, int(prediction.item())].item()*100)
    return [int(prediction.item()), confidence]


# To validate the dataset
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    # To get number of videos
    def __len__(self):
        return len(self.video_names)

    # To get number of frames
    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if (len(frames) == self.count):
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    # To extract number of frames
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image


def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    path_to_videos = [videoPath]

    video_dataset = validation_dataset(
        path_to_videos, sequence_length=20, transform=train_transforms)
    # use this command for gpu
    # model = Model(2).cuda()
    # model = Model(2)
    # path_to_model = 'model_93_acc_100_frames_celeb_FF_data.pt'
    # model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    # model.eval()
    for i in range(0, len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model, video_dataset[i], './')
        if prediction[0] == 1:
            print("REAL")
        else:
            print("FAKE")
    return prediction


# Define a function to preprocess images for the model


def preprocess_image(frame):
    frame = cv2.resize(frame, (128, 128))
    x = np.expand_dims(frame, axis=0)
    x = preprocess_input(x)
    return x

# Define a function to predict with the model


def predict_fn(images):
    preds = deepfake_model.predict(images)
    return preds


def deepfake_xai(video_path):
   # Load an example video
    cap = cv2.VideoCapture(video_path)
    # Create LIME explainer for image classification
    explainer = lime_image.LimeImageExplainer()

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total : ", total_frames)

    # Select 5 random frames
    selected_indices = random.sample(range(total_frames), 5)

    # Ensure the output directory exists
    output_dir = 'static'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Assuming selected_indices and other variables are defined earlier in your code
    for idx, selected_index in enumerate(selected_indices, start=1):
        print("idx : ", idx)
        # Seek to the selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_index)
        ret, frame = cap.read()
        if ret:
            # Preprocess the frame
            frame = cv2.resize(frame, (128, 128))
            x = np.expand_dims(frame, axis=0)
            x = preprocess_input(x)

            # Explain the prediction
            explanation = explainer.explain_instance(
                x[0], predict_fn, top_labels=5, hide_color=0, num_samples=200)

            # Show explanation
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            image_path = os.path.join(output_dir, f'{idx}.png')
            print("image path : ", image_path)
            # Open the image with the default image viewer
            # if os.name == 'nt':  # Windows
            #     os.startfile(image_path)
            plt.savefig(image_path, format='png')

    print("xai done")
    # Release the video capture
    cap.release()
    return


@app.route('/', methods=['POST', 'GET'])
def homepage():
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('index1.html', data="hello world")
    if request.method == 'POST':
        video = request.files['video']
        print(video.filename)
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(
            app.config['UPLOAD_FOLDER'], video_filename))
        video_path = "Uploaded_Files/" + video_filename
        prediction = detectFakeVideo(video_path)
        print(prediction)

        deepfake_xai(video_path)

        if prediction[0] == 0:
            output = "FAKE"
        else:
            output = "REAL"
        confidence = prediction[1]
        data = {'output': output, 'confidence': confidence, }
        # data = {'output': 'output', 'confidence': 'confidence',
        #         "explanations": 'explanations'}
        # data = json.dumps(data)
        os.remove(video_path)
        print("data : ", data)
        return jsonify(data)


app.run(debug=True, port=3000)
