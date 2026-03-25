
import torch
import pandas as pd

def predict(model, dataloader):

    preds = []

    model.eval()

    with torch.no_grad():

        for x in dataloader:
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())

    return pd.DataFrame({
        "prediction": preds
    })
