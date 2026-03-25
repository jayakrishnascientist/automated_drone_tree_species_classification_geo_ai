# QGIS + QField Workflow for Tree Crown Ground-Truthing

## Overview
This project provides a structured workflow for ground-truthing tree crown polygons using QGIS and QField. It integrates remote sensing outputs, tree crown species classification, etc., with field-based validation to create accurate, labelled ecological datasets.

## Workflow Summary
1. Data Preparation  
2. QGIS Project Setup  
3. Attribute Configuration  
4. Visualization  
5. QField Cloud Deployment  
6. Field Data Collection  
7. Data Synchronization  

## 1. Data Preparation

### Steps 1
- Compress orthomosaic (OM) raster without altering georeferencing  
- Convert crown polygons from .geojson to .gpkg  
- Ensure CRS consistency across all datasets  

Efficient field workflows require lightweight datasets. Compression reduces file size while preserving spatial accuracy. GeoPackage (.gpkg) is preferred because it is faster, more stable, and fully supported in mobile environments like QField.

## 2. QGIS Project Setup

<img width="600" height="600" alt="step1" src="https://github.com/user-attachments/assets/36f2e2d3-2f96-4140-b342-c3b49f128c3c" />

### Steps 2
- Create a new QGIS project  
- Set project CRS  
- Load orthomosaic (OM) and crown polygons (GPKG)  
- Verify spatial alignment  

QGIS acts as the central environment where spatial layers are integrated. Correct CRS alignment ensures that all datasets overlay accurately, which is critical for reliable field validation.

## 3. Attribute Schema Configuration

### Steps 3
### step 3 (a) adding fields
<img width="600" height="600" alt="step2" src="https://github.com/user-attachments/assets/6e4a3036-dfd4-4db7-845c-8c6bca79915e" />
### Step 3 (b) adding atributes
<img width="600" height="600" alt="step 3" src="https://github.com/user-attachments/assets/f9516521-654a-4b51-bef9-3c22a013c36d" />

Add fields to crown polygon layer:

- crown_id (Integer)  
- species (String)  
- description (String)  
- photo (String)  
- status (String)  
- tree_type (String)  
- health (String)  

### Form Customization
Set widget types in Attribute Form:
### Step 3 (c)  set widgets

<img width="600" height="600" alt="Screenshot 2026-03-18 at 2 00 35 PM" src="https://github.com/user-attachments/assets/727c396a-8662-4113-8b58-6b77f7a52128" />

- species → Value Map (species list)  
- status → pending / completed  
- tree_type → acacia / non-acacia  
- health → flowering / leaf shed / full canopy / snag-dead  
- photo → Attachment  

Standardized attribute forms reduce human error during field data collection. Value maps enforce controlled vocabularies, ensuring consistent and analyzable datasets.

## 4. Visualization and Styling
### step 4 Apply rule-based 

<img width="600" height="600" alt="Screenshot 2026-03-18 at 1 59 06 PM" src="https://github.com/user-attachments/assets/a99ece06-c1e0-4927-9881-13272a3648cf" />

### Steps
- Set polygon fill to transparent  
- Enable boundary outline  
- Apply rule-based styling:
  - Completed → Green  
  - Pending → Red  

Visual cues improve field efficiency. Color-coded status helps users quickly identify which trees are already labeled and which require attention.

## 5. QField Cloud Deployment
### step 5 (a) after sync project from QGIS  and  check weather its updated in the in cloud project 

<img width="800" height="800" alt="Screenshot 2026-03-18 at 2 02 57 PM" src="https://github.com/user-attachments/assets/052ee86a-a26e-49c5-8767-92b89c347170" />

### step 5 (a) 


### Steps
- Create QField Cloud account  
- Install QFieldCloud plugin in QGIS  
- Upload project to cloud  
- Ensure all paths are relative  

Cloud synchronization enables seamless data transfer between desktop (QGIS) and mobile (QField), supporting collaborative and real-time workflows.

## 6. Field Data Collection (QField)

### Steps
- Install QField app  
- Login and download project  
- Use GPS to locate crowns  
- Update attributes (species, status, health, tree_type)  
- Capture photos  
- Save edits  

QField enables in-situ data validation. Integrating GPS with spatial layers allows precise mapping of ecological attributes directly in the field.

## 7. Collaboration
### step 1 set role
<img width="500" height="500" alt="Screenshot 2026-03-18 at 2 03 13 PM" src="https://github.com/user-attachments/assets/3f677a63-0ee6-4aaf-83a2-46a86fa672e7" />

### Steps
- Add contributors in QField Cloud  
- Assign roles (Editor)  
- Share project access  

Collaborative editing allows multiple users to collect data simultaneously, significantly speeding up large-scale ecological surveys.

## 8. Data Synchronization

### step 8 Always do push changes on mobile
<img width="300" height="400" alt="IMG_0752" src="https://github.com/user-attachments/assets/2870108d-89e5-4a43-8ae1-5b683d6fb814" />


### Steps
- Push updates from QField to Cloud  
- Sync project back to QGIS  
- Or download updated GPKG  

Bidirectional synchronization ensures that field updates are reflected in the master dataset, enabling continuous refinement and analysis.

## Best Practices

- Always maintain CRS consistency  
- Use GeoPackage instead of GeoJSON  
- Keep backups before syncing  
- Use default values (e.g., status = pending)  
- Optimize raster size for mobile performance  

## Output
## download the gpkg file for further analysis
<img width="760" height="462" alt="Screenshot 2026-03-18 at 2 03 38 PM" src="https://github.com/user-attachments/assets/bf196aa6-22d6-408e-9a7b-a4712575c9d1" />


- Fully labeled tree crown dataset  
- Georeferenced images linked to crowns  
- Ready for ecological analysis, machine learning training, and reporting

## Use Case

This workflow is suitable for forest monitoring, species classification, UAV-based ecological surveys, and ground-truth data collection.
