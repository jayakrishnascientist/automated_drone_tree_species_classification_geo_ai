
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from utils import read_tif

def extract_resnet_features(image_folder, output_dir, batch_size=32, img_size=224):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    image_paths = sorted(
        glob.glob(os.path.join(image_folder,"*.tif")) +
        glob.glob(os.path.join(image_folder,"*.tiff"))
    )

    features_list = []
    names_list = []

    with torch.no_grad():

        for i in tqdm(range(0,len(image_paths),batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_imgs = []

            for p in batch_paths:
                img = transform(read_tif(p))
                batch_imgs.append(img)
                names_list.append(os.path.basename(p))

            batch_tensor = torch.stack(batch_imgs).to(device)

            feats = model(batch_tensor)
            feats = feats.view(feats.size(0), -1)
            feats = torch.nn.functional.normalize(feats,p=2,dim=1)

            features_list.append(feats.cpu().numpy())

    features = np.vstack(features_list)

    np.save(os.path.join(output_dir,"resnet_features.npy"),features)

    df = pd.DataFrame(features)
    df.insert(0,"image_name",names_list)
    df.to_csv(os.path.join(output_dir,"resnet_features.csv"),index=False)
