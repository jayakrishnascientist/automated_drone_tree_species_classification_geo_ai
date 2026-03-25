# 🚀 Quick Start Guide

## For Mac Users with Visual Studio Code

### 1️⃣ Install & Setup (5 minutes)

```bash
# Open Terminal and run these commands:

# Go to your Desktop (or any folder you want)
cd ~/Desktop

# Clone your repo (replace with your GitHub URL)
git clone https://github.com/yourusername/tree-crown-pipeline.git
cd tree-crown-pipeline

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Open in VS Code

```bash
# Open VS Code from terminal
code .
```

Then:
1. Press `Cmd+Shift+P`
2. Type: "Python: Select Interpreter"
3. Choose: `./venv/bin/python`

### 3️⃣ Configure Your Data

Open `tree_crown_pipeline.py` and edit lines 98-105:

```python
class Config:
    # Update these paths:
    ORTHO_FOLDER = '/Users/yourname/data/orthomosaics'
    POLY_FOLDER = '/Users/yourname/data/polygons'
    STEP1_OUTPUT = '/Users/yourname/outputs/step1_output'
    CHOSEN_K = 6  # Update after Step 1
```

💡 **Tip**: Drag folders into Terminal to get the full path!

### 4️⃣ Run Step 1

In VS Code terminal (Ctrl+`):

```bash
# Activate environment
source venv/bin/activate

# Run Step 1
python tree_crown_pipeline.py --step 1
```

⏱️ **Time**: 10-60 minutes depending on your data size

### 5️⃣ Choose k and Assign Species

1. **Browse cluster folders**: 
   - Go to: `step1_output/clustering/k{k}/cluster_{i}/`
   - Look at the tree images in each cluster

2. **Check metrics**:
   - Open: `clustering/k_selection.png`
   - Look at rank 1 in: `k_recommendation_table.csv`

3. **Fill species CSV**:
   - Open: `clustering/k6_cluster_species_map.csv` (use your chosen k)
   - Fill the `species` column:
   
   ```csv
   cluster,cluster_folder,species,notes
   0,cluster_0,acacia,Clear acacia pattern
   1,cluster_1,eucalyptus,Broad leaves
   2,cluster_2,pine,Needle texture
   ```

4. **Update config**:
   ```python
   CHOSEN_K = 6  # Your chosen k
   ```

### 6️⃣ Run Step 2

```bash
python tree_crown_pipeline.py --step 2
```

✅ **Output**: Species assignments in `step2_output/`

### 7️⃣ Export to Google Earth

```bash
python tree_crown_pipeline.py --step 4
```

📍 **Open in Google Earth**:
1. Go to [earth.google.com](https://earth.google.com)
2. Click **Projects** → **Import KML**
3. Upload: `step4_output/species_map.kmz`

---

## Single Command (All Steps)

After configuring paths and filling species CSV:

```bash
python tree_crown_pipeline.py --step all
```

---

## Troubleshooting

### "Out of memory"
```python
BATCH_SIZE = 8  # Reduce this in Config
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### Slow processing
```python
MODEL_NAME = 'vit_small_patch14_dinov2.lvd142m'  # Use smaller model
```

---

## File Structure

```
tree-crown-pipeline/
├── tree_crown_pipeline.py    # ← Main script
├── requirements.txt           # ← Dependencies
├── README.md                  # ← Full docs
├── SETUP_MAC.md              # ← Mac setup guide
├── QUICKSTART.md             # ← This file
└── config_example.py          # ← Config template
```

---

## Commands Cheat Sheet

```bash
# Activate environment
source venv/bin/activate

# Run individual steps
python tree_crown_pipeline.py --step 1  # Clustering
python tree_crown_pipeline.py --step 2  # Species assignment
python tree_crown_pipeline.py --step 3  # Validation (optional)
python tree_crown_pipeline.py --step 4  # KMZ export

# Run all steps
python tree_crown_pipeline.py --step all

# Deactivate environment
deactivate
```

---

## Need Help?

- Read **README.md** for full documentation
- Read **SETUP_MAC.md** for Mac-specific setup
- Check your config paths in `tree_crown_pipeline.py`
- Make sure your data format matches (see README)

---

Good luck! 🌳✨
