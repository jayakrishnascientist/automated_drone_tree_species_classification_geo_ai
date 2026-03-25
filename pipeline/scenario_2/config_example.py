# Example Configuration File
# Copy this to config.py and edit your paths

class Config:
    """Pipeline configuration"""
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: CLUSTERING
    # ═══════════════════════════════════════════════════════════════
    
    # Input data folders
    ORTHO_FOLDER = '/Users/yourname/Desktop/data/orthomosaics'
    POLY_FOLDER = '/Users/yourname/Desktop/data/polygons'
    
    # Output folder for Step 1
    STEP1_OUTPUT = '/Users/yourname/Desktop/outputs/step1_output'
    
    # K-means parameters
    K_LIST = [2, 4, 6, 8, 10, 12]  # List of k values to test
    
    # DINOv2 model selection
    # Options:
    #   'vit_small_patch14_dinov2.lvd142m'  - Fast, less memory (~22M params)
    #   'vit_base_patch14_dinov2.lvd142m'   - Recommended (~86M params)
    #   'vit_large_patch14_dinov2.lvd142m'  - Best quality (~304M params)
    MODEL_NAME = 'vit_base_patch14_dinov2.lvd142m'
    
    # Image processing
    IMG_SIZE = 224  # Image size for DINOv2 (don't change unless needed)
    BATCH_SIZE = 32  # Reduce to 16, 8, or 4 if out of memory
    
    # Dimensionality reduction
    PCA_COMPONENTS = 50  # Number of PCA components (None to skip PCA)
    
    # Disk space management
    COPY_TO_CLUSTER_FOLDERS = True  # Set False to save disk space
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: SPECIES ASSIGNMENT
    # ═══════════════════════════════════════════════════════════════
    
    # Chosen k value (UPDATE THIS after Step 1 inspection)
    CHOSEN_K = 6  # Replace with your chosen k
    
    # Output folder for Step 2
    STEP2_OUTPUT = '/Users/yourname/Desktop/outputs/step2_output'
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: VALIDATION (OPTIONAL)
    # ═══════════════════════════════════════════════════════════════
    
    # Path to your ground truth CSV
    # Format: image_name, species
    # Example:
    #   site1_000.tif,acacia
    #   site1_001.tif,eucalyptus
    GROUND_TRUTH_CSV = '/Users/yourname/Desktop/data/ground_truth.csv'
    
    # Validation output folder (None = auto-set)
    STEP3_VALIDATION_OUTPUT = None
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: KMZ EXPORT
    # ═══════════════════════════════════════════════════════════════
    
    # Output folder for Step 4
    STEP4_OUTPUT = '/Users/yourname/Desktop/outputs/step4_output'
    
    # Source EPSG code of your polygon GeoJSONs
    # Common codes:
    #   32643 = WGS 84 / UTM zone 43N (India)
    #   32644 = WGS 84 / UTM zone 44N (India)
    #   32642 = WGS 84 / UTM zone 42N (India)
    #   4326  = WGS 84 (lat/lon)
    SOURCE_EPSG = 32643
    
    # KML color palette (AABBGGRR format)
    # Each species gets one color in order
    COLOR_PALETTE = [
        '990000ff',  # Blue
        '9900ff00',  # Green
        '99ff0000',  # Red
        '9900ffff',  # Yellow
        '99ff00ff',  # Magenta
        '99ff8800',  # Orange
        '9900ffff',  # Cyan
        '99ffffff',  # White
    ]


# ═══════════════════════════════════════════════════════════════════
# ADVANCED SETTINGS (rarely need to change)
# ═══════════════════════════════════════════════════════════════════

class AdvancedConfig:
    """Advanced settings for expert users"""
    
    # K-means parameters
    KMEANS_N_INIT = 10  # Number of K-means initializations
    KMEANS_RANDOM_STATE = 42  # Random seed for reproducibility
    
    # t-SNE parameters
    TSNE_PERPLEXITY = 30  # t-SNE perplexity (5-50)
    TSNE_RANDOM_STATE = 42  # Random seed
    
    # Feature extraction
    NORMALIZE_FEATURES = True  # L2 normalization
    STANDARDIZE_FEATURES = True  # StandardScaler before PCA
    
    # Silhouette score sampling
    SILHOUETTE_SAMPLE_SIZE = 5000  # Max samples for silhouette (speed)
    
    # Plot settings
    PLOT_DPI = 150  # Plot resolution
    PLOT_STYLE = 'default'  # Matplotlib style
