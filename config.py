# =============================================================================
# config_example.py
# =============================================================================
# Copy this file to config.py and edit the paths for your data.
#
# Usage:
#   cp config_example.py config.py
#   # edit config.py with your paths
#   python tree_crown_pipeline.py --step 1 --config config.py
#   python tree_crown_pipeline.py --step 2 --config config.py
#   python tree_crown_pipeline.py --step 3 --config config.py   # optional
#   python tree_crown_pipeline.py --step 4 --config config.py
# =============================================================================


class Config:
    """Pipeline configuration — edit paths below, then run with --config config.py"""

    # =========================================================================
    # STEP 1 — Crown cropping, DINOv2 feature extraction, K-Means clustering
    # =========================================================================

    # Folder containing orthomosaic GeoTIFFs (.tif files)
    ORTHO_FOLDER = '/Users/Shared/Files From d.localized/Guide_IITD/data_set_sanjayvan/Data/sanjay_van/Sanjay_Van_Drone/spot1_8_10/final key/git_clone/scenario_2/ortho'

    # Folder containing crown polygon GeoJSONs (one .geojson per site)
    POLY_FOLDER = '/Users/Shared/Files From d.localized/Guide_IITD/data_set_sanjayvan/Data/sanjay_van/Sanjay_Van_Drone/spot1_8_10/final key/git_clone/scenario_2/geojson'

    # Output folder for all Step 1 results
    STEP1_OUTPUT = '/Users/Shared/Files From d.localized/Guide_IITD/data_set_sanjayvan/Data/sanjay_van/Sanjay_Van_Drone/spot1_8_10/final key/git_clone/scenario_2/step1_output'


    # K values to evaluate (adjust range based on expected number of species)
    K_LIST = [2, 4, 6, 8, 10, 12]

    # DINOv2 backbone — choose based on available memory:
    #   'vit_small_patch14_dinov2.lvd142m'  →  faster, ~22M params, less memory
    #   'vit_base_patch14_dinov2.lvd142m'   →  recommended, ~86M params
    #   'vit_large_patch14_dinov2.lvd142m'  →  best quality, ~304M params, slow
    MODEL_NAME = 'vit_base_patch14_dinov2.lvd142m'

    IMG_SIZE   = 224   # DINOv2 input size — do not change
    BATCH_SIZE = 32    # Reduce to 16, 8, or 4 if you get out-of-memory errors

    # Number of PCA components before clustering (set None to skip PCA)
    PCA_COMPONENTS = 50

    # Copy crown TIFFs into per-cluster subfolders for visual inspection
    # Set False to save disk space (cluster CSVs are still written)
    COPY_TO_CLUSTER_FOLDERS = True

    # =========================================================================
    # STEP 2 — Species assignment (run AFTER Step 1 and visual inspection)
    # =========================================================================

    # The k value you chose after inspecting Step 1 cluster folders and plots
    # Update this value before running Step 2
    CHOSEN_K = 6

    # Output folder for Step 2 results (crown_master.csv, species folders, etc.)
    STEP2_OUTPUT = '/path/to/your/step2_output'

    # =========================================================================
    # STEP 3 — Validation (OPTIONAL — can skip to Step 4)
    # =========================================================================

    # Path to your ground truth CSV.
    # Required columns: image_name (e.g. s1_tree_006.tif) + label/species
    # Column names are auto-detected — any header works.
    #
    # Example CSV format:
    #   image_name,label
    #   s1_tree_006.tif,acacia
    #   s1_tree_009.tif,non_acacia
    #   s1_tree_013.tif,acacia
    #
    # Set to None or empty string to skip Step 3 gracefully.
    GROUND_TRUTH_CSV = '/path/to/your/ground_truth.csv'

    # Output folder for validation results (None = auto: step2_output/step3_validation)
    STEP3_VALIDATION_OUTPUT = None

    # =========================================================================
    # STEP 4 — KMZ export for Google Earth
    # =========================================================================

    # Output folder for the final KMZ file
    STEP4_OUTPUT = '/path/to/your/step4_output'

    # EPSG code of your crown polygon GeoJSONs
    # Common codes for India:
    #   32642 = WGS 84 / UTM zone 42N
    #   32643 = WGS 84 / UTM zone 43N  (Delhi / Sanjay Van)
    #   32644 = WGS 84 / UTM zone 44N
    #   4326  = WGS 84 geographic (lat/lon)
    SOURCE_EPSG = 32643

    # KML polygon fill colors — one per species, in the order they appear.
    # Format: AABBGGRR  (alpha, blue, green, red — KML standard)
    # Add more entries if you have more than 8 species.
    COLOR_PALETTE = [
        '990000ff',  # blue
        '9900ff00',  # green
        '99ff0000',  # red
        '9900ffff',  # yellow
        '99ff00ff',  # magenta
        '99ff8800',  # orange
        '9900ffff',  # cyan
        '99ffffff',  # white
    ]


# =============================================================================
# ADVANCED SETTINGS (rarely need to change)
# =============================================================================

class AdvancedConfig:
    """Advanced settings for expert users.
    These are not read by the main pipeline — modify tree_crown_pipeline.py
    directly if you need to change these values.
    """

    # K-Means
    KMEANS_N_INIT      = 10   # Number of K-Means initializations
    KMEANS_RANDOM_STATE = 42  # Random seed for reproducibility

    # t-SNE
    TSNE_PERPLEXITY    = 30   # Perplexity parameter (5–50)
    TSNE_RANDOM_STATE  = 42   # Random seed

    # Feature normalization
    NORMALIZE_FEATURES   = True   # L2 normalization of DINOv2 output
    STANDARDIZE_FEATURES = True   # StandardScaler before PCA

    # Silhouette score
    SILHOUETTE_SAMPLE_SIZE = 5000  # Max samples used (for speed)

    # Plot output
    PLOT_DPI   = 150       # Resolution of saved plots
    PLOT_STYLE = 'default' # Matplotlib style
