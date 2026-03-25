
import os
import pandas as pd

def create_label_sheet(label_pool_dir, output_csv):

    classes = ["acacia", "non_acacia"]
    data = []

    for cls in classes:
        class_dir = os.path.join(label_pool_dir, cls)

        if not os.path.exists(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".tif",".tiff",".png",".jpg",".jpeg")):
                data.append({
                    "image_name": fname,
                    "label": cls
                })

    df = pd.DataFrame(data)
    df = df.sort_values("image_name").reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print("Label sheet created:", output_csv)
