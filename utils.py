import os
import torch
import torchvision.transforms as trfm
from PIL import Image
import models.facenet
import cv2
from statistics import mean
import numpy as np




def load_weight(model, model_name: str, device):

    parameters = os.path.join('var/www/weights/', model_name + '.pth')

    if model_name.split("_")[-1] == "Quantized":
        model.eval()
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        torch.backends.quantized.engine = 'fbgemm'
        torch.quantization.convert(model, inplace=True)
    
    checkpoint = torch.load(parameters, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        model.module.load_state_dict(checkpoint['state_dict'])
    
    model.eval()

    return model


def detect_face(image):
    PADDING = 0
    face_in_frame = image
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    verified = FACE_CASCADE.detectMultiScale(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY), 1.3, 5)

    for (x, y, w, h) in verified:
        x1 = x - PADDING
        y1 = y - PADDING
        x2 = x + w + PADDING
        y2 = y + h + PADDING
        
        face_in_frame = cv2_image[y1:y2, x1:x2]
        face_in_frame = Image.fromarray(cv2.cvtColor(face_in_frame, cv2.COLOR_BGR2RGB))

    return face_in_frame



def _is_same(img1, img2,model,trfrm,threshold):
        model.eval()
        with torch.no_grad():
            embed1 = model(trfrm(img1).unsqueeze(0))
            embed2 = model(trfrm(img2).unsqueeze(0))
        euclidean_distance = torch.nn.functional.pairwise_distance(embed1, embed2).item()
        
        # ave_buffer.append(euclidean_distance)
        # euclidean_distance = mean(ave_buffer)

        return euclidean_distance, euclidean_distance <= threshold