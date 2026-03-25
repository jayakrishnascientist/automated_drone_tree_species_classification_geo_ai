
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from utils import read_tif

def extract_vit_features(image_folder, output_dir, batch_size=16, img_size=224):

    os.makedirs(output_dir, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model.to(device).eval()

    transform=transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    img_paths=sorted(glob.glob(os.path.join(image_folder,"*.tif")))

    all_features=[]
    all_names=[]

    for i in tqdm(range(0,len(img_paths),batch_size)):

        batch_paths=img_paths[i:i+batch_size]
        batch_imgs=[]
        names=[]

        for p in batch_paths:
            batch_imgs.append(transform(read_tif(p)))
            names.append(os.path.basename(p))

        batch_tensor=torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            feats=model(batch_tensor)

        feats=F.normalize(feats,p=2,dim=1)

        all_features.append(feats.cpu().numpy())
        all_names.extend(names)

    features_np=np.vstack(all_features)

    np.save(os.path.join(output_dir,"vit_features.npy"),features_np)

    pd.DataFrame({"image_name":all_names}).to_csv(
        os.path.join(output_dir,"vit_features.csv"), index=False)
