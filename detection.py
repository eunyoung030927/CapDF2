import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

class Detection:
    def __init__(self):
        self.model_path = r".\results\ResNet2_test.pt"
        self.face_model_path = r".\face_crop_file\haarcascade_frontalface_default.xml"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.CenterCrop(128), # 작은 크기로 축소 -> 화질 낮추기
            transforms.ToTensor(),  
            transforms.Normalize((0.5981726, 0.430156, 0.3900312), (0.1940932, 0.1525563, 0.14132583)) 
        ])
        self.model = self.load_model() # 모델 준비 

    def load_model(self):
        model = torch.load(self.model_path, map_location=self.DEVICE)
        model.eval()
        return model
    
    def detect(self, img):
        # img_path = r"D:\FOM_deepfake2\test\image\00002\00027.jpg"
        # img = Image.open(img_path)
        # img = self.transform(img).unsqueeze(0).to(self.DEVICE) # (1,3,128,128)
        img_new = self.prepare_img(img) # 이미지 준비

        with torch.no_grad():
            y_hat = self.model(img_new)
            prob = torch.sigmoid(y_hat).item()
            pred = (prob >= 0.5)

        return prob, pred
    
    def prepare_img(self, img_pil):
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # pil -> cv2

        face_cascade = cv2.CascadeClassifier(self.face_model_path)
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0: # 얼굴이 탐지되지 않으면
            img_complete = self.transform(img_pil).unsqueeze(0).to(self.DEVICE) # (1,3,128,128)
            return img_complete # 입력된 이미지에 transforms만 # (1,3,128,128)
        
        else: 
            for (x,y,w,h) in faces:
                margin = int(0.7 * h)  # 얼굴 높이 50%를 마진으로
                x_start = max(0, x - margin)  # 이미지 벗어나지 않도록
                y_start = max(0, y - margin)
                x_end = min(img_cv2.shape[1], x + w + margin)
                y_end = min(img_cv2.shape[0], y + h + margin)

                img_cropped = img_cv2[y_start:y_end, x_start:x_end]
                img_new = cv2.resize(img_cropped, (128,128))

            cropped_pil = Image.fromarray(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)) # cv2 -> pil
            img_complete = self.transform(cropped_pil).unsqueeze(0).to(self.DEVICE) # (1,3,128,128)

            return img_complete

