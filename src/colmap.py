import os
import subprocess
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('colmap_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def run_command(command):
    """Run a shell command and handle errors"""
    logger.info(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"Error: {e.stderr}")
        return False

def ensure_directory(path):
    """Ensure a directory exists, create it if it doesn't"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

def run_colmap_pipeline(dataset_path):
    """Run the full COLMAP pipeline with texturing and error handling"""
    # Define paths
    database_path = os.path.join(dataset_path, "database.db")
    images_path = os.path.join(dataset_path, "images")
    sparse_path = os.path.join(dataset_path, "sparse")
    dense_path = os.path.join(dataset_path, "dense")
    fused_ply = os.path.join(dense_path, "fused.ply")
    meshed_poisson = os.path.join(dense_path, "meshed-poisson.ply")
    meshed_delaunay = os.path.join(dense_path, "meshed-delaunay.ply")
    
    # Validate input directory
    if not os.path.isdir(images_path):
        logger.error(f"Images directory not found: {images_path}")
        return False
    
    # Create necessary directories
    if not ensure_directory(sparse_path) or not ensure_directory(dense_path):
        return False
    
    # Define the pipeline steps
    steps = [
        {
            "name": "Feature Extraction",
            "command": f"colmap feature_extractor --database_path {database_path} --image_path {images_path}"
        },
        {
            "name": "Feature Matching",
            "command": f"colmap exhaustive_matcher --database_path {database_path}"
        },
        {
            "name": "Sparse Reconstruction",
            "command": f"colmap mapper --database_path {database_path} --image_path {images_path} --output_path {sparse_path}"
        },
        {
            "name": "Image Undistortion",
            "command": f"colmap image_undistorter --image_path {images_path} --input_path {sparse_path}/0 --output_path {dense_path} --output_type COLMAP --max_image_size 2000"
        },
        {
            "name": "Stereo Matching",
            "command": f"colmap patch_match_stereo --workspace_path {dense_path} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true"
        },
        {
            "name": "Stereo Fusion",
            "command": f"colmap stereo_fusion --workspace_path {dense_path} --workspace_format COLMAP --input_type geometric --output_path {fused_ply}"
        },
        {
            "name": "Poisson Surface Reconstruction",
            "command": f"colmap poisson_mesher --input_path {fused_ply} --output_path {meshed_poisson}"
        },
        {
            "name": "Delaunay Surface Reconstruction",
            "command": f"colmap delaunay_mesher --input_path {dense_path} --output_path {meshed_delaunay}"
        }
    ]
    
    # Run each step, handling errors
    for step in steps:
        logger.info(f"Starting {step['name']}...")
        if not run_command(step["command"]):
            logger.error(f"Failed during {step['name']} step")
            return False
        logger.info(f"Completed {step['name']}")
    
    logger.info("COLMAP pipeline completed successfully")
    return True

if __name__ == "__main__":
    try:
        dataset_path = "data/NissanMurano/colmap"
        logger.info(f"Starting COLMAP pipeline for dataset: {dataset_path}")
        success = run_colmap_pipeline(dataset_path)
        if success:
            logger.info("Pipeline completed successfully")
        else:
            logger.error("Pipeline failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
