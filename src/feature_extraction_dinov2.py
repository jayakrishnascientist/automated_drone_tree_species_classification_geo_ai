
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from utils import read_tif

def extract_dinov2_features(image_folder, output_dir, batch_size=16, img_size=224):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
        img_size=224
    ).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    image_paths = sorted(glob.glob(os.path.join(image_folder,"*.tif")))

    features_list=[]
    names_list=[]

    with torch.no_grad():

        for i in tqdm(range(0,len(image_paths),batch_size)):
            batch_paths=image_paths[i:i+batch_size]
            imgs=[]

            for p in batch_paths:
                imgs.append(transform(read_tif(p)))
                names_list.append(os.path.basename(p))

            batch=torch.stack(imgs).to(device)

            feats=model(batch)
            feats=F.normalize(feats,p=2,dim=1)

            features_list.append(feats.cpu().numpy())

    features=np.vstack(features_list)

    np.save(os.path.join(output_dir,"dinov2_features.npy"),features)

    pd.DataFrame({"image_name":names_list}).to_csv(
        os.path.join(output_dir,"dinov2_features.csv"),index=False)
