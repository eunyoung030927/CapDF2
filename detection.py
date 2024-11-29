import torch
from torchvision import transforms
from PIL import Image
import matplotlib as plt
import os

class Detection:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 절대 경로
        self.model_path = os.path.join(base_path, "results", "ResNet2_test.pt")
        # self.model_path = r"results\ResNet2_test.pt"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),  
            # transforms.CenterCrop(196), # 작은 크기로 축소 -> 화질 낮추기
            transforms.Normalize((0.5981726, 0.430156, 0.3900312), (0.1940932, 0.1525563, 0.14132583)) 
        ])
        self.model = self.load_model()

    def load_model(self):
        model = torch.load(self.model_path, map_location=self.DEVICE)
        model.eval()
        return model
    
    def detect(self, img):
        # img_path = r"D:\FOM_deepfake2\test\image\00002\00027.jpg"
        # img = Image.open(img_path)
        img = self.transform(img).unsqueeze(0).to(self.DEVICE) # (1,3,128,128)

        with torch.no_grad():
            y_hat = self.model(img)
            prob = torch.sigmoid(y_hat).item()
            pred = (prob >= 0.5)

        return prob, pred

