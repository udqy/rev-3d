import os
import cv2
import numpy as np
import requests
from segment_anything import sam_model_registry, SamPredictor
import torch

def ensure_directory(directory_path):
    """
    Ensure the specified directory exists.
    
    Args:
        directory_path (str): Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def download_sam_model(checkpoint_path):
    """
    Download the SAM model if not already downloaded.
    
    Args:
        checkpoint_path (str): Path where the model should be saved
    """
    if not os.path.exists(checkpoint_path):
        print("Downloading SAM model weights...")
        ensure_directory(os.path.dirname(checkpoint_path))
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        response = requests.get(url)
        with open(checkpoint_path, 'wb') as f:
            f.write(response.content)
        print("Download complete!")

def initialize_sam_model(checkpoint_path):
    """
    Initialize the SAM model.
    
    Args:
        checkpoint_path (str): Path to the SAM model checkpoint
        
    Returns:
        SamPredictor: Initialized SAM predictor
    """
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)

def apply_sam(image, predictor):
    """
    Apply SAM to remove the background from an image.
    
    Args:
        image (numpy.ndarray): Input image
        predictor (SamPredictor): Initialized SAM predictor
        
    Returns:
        numpy.ndarray: Image with background removed
        numpy.ndarray: Binary mask
    """
    # Convert to RGB if needed
    if image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image
        
    predictor.set_image(img_rgb)
    
    # Select points in the center of the car
    h, w = img_rgb.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Create input points (center of the image where the car is)
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([1])  # 1 means foreground
    
    # Get the mask from SAM
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    
    # Choose the mask with the highest score
    mask_idx = np.argmax(scores)
    mask = masks[mask_idx]
    
    # Apply the mask to the image
    result = img_rgb.copy()
    result[~mask] = [0, 0, 0]
    
    return result, mask

def evaluate_frame_quality(mask):
    """
    Evaluate the quality of a frame based on the mask.
    
    Args:
        mask (numpy.ndarray): Binary mask from SAM
        
    Returns:
        float: Quality score between 0 and 1
        dict: Detailed quality metrics
    """
    # Get mask properties
    y_indices, x_indices = np.where(mask)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return 0.0, {"coverage": 0, "distribution": 0, "shape": 0, "error": "Empty mask"}
    
    # 1. Coverage - what percentage of the frame is the object
    coverage = np.sum(mask) / mask.size
    
    # 2. Distribution - how well distributed the object is
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min
    
    # Frame dimensions
    frame_height, frame_width = mask.shape
    
    # Calculate aspect ratio of the masked object
    if height == 0:  # Avoid division by zero
        aspect_ratio = 0
    else:
        aspect_ratio = width / height
    
    # Calculate distribution score based on how much of the frame is used
    width_ratio = width / frame_width
    height_ratio = height / frame_height
    distribution = (width_ratio + height_ratio) / 2
    
    # 3. Shape evaluation - penalize extreme aspect ratios
    # For a car, we expect aspect ratio around 1.5-2.5 (width/height)
    ideal_aspect_min, ideal_aspect_max = 1.0, 3.0
    
    if aspect_ratio < ideal_aspect_min:
        # Too tall and narrow (like just a door)
        shape_score = aspect_ratio / ideal_aspect_min
    elif aspect_ratio > ideal_aspect_max:
        # Too wide and short (like just a hood or roof)
        shape_score = ideal_aspect_max / aspect_ratio
    else:
        # Within ideal range
        shape_score = 1.0
    
    # 4. Fill density - how dense is the object within its bounding box
    bbox_area = width * height
    if bbox_area == 0:  # Avoid division by zero
        fill_density = 0
    else:
        fill_density = np.sum(mask) / bbox_area
    
    # 5. Edge detection - detect if only a thin strip is visible
    edge_threshold = 0.2  # Threshold for edge detection
    is_edge_only = (width_ratio < edge_threshold or height_ratio < edge_threshold)
    edge_penalty = 0.5 if is_edge_only else 1.0
    
    # Calculate final quality score - weighted combination of factors
    quality_score = (
        0.25 * coverage +
        0.25 * distribution +
        0.25 * shape_score +
        0.25 * fill_density
    ) * edge_penalty
    
    # Cap between 0 and 1
    quality_score = max(0.0, min(1.0, quality_score))
    
    # Return detailed metrics for debugging
    metrics = {
        "coverage": coverage,
        "distribution": distribution,
        "shape_score": shape_score,
        "fill_density": fill_density,
        "aspect_ratio": aspect_ratio,
        "edge_penalty": edge_penalty
    }
    
    return quality_score, metrics

def extract_and_process_frames(video_path, output_dir, num_frames=60, quality_threshold=0.3, max_extra_frames=20):
    """
    Extract frames from video, process with SAM, and save results.
    Filter out poor quality frames.
    
    Args:
        video_path (str): Path to the input video
        output_dir (str): Directory to save processed frames
        num_frames (int): Target number of frames to extract
        quality_threshold (float): Minimum quality score to keep a frame
        max_extra_frames (int): Maximum additional frames to extract to replace bad ones
    """
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Initialize SAM model
    checkpoint_path = "../models/sam_vit_h_4b8939.pth"
    download_sam_model(checkpoint_path)
    predictor = initialize_sam_model(checkpoint_path)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate which frames to extract initially
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    good_frames = []
    rejected_frames = []
    
    print(f"Processing initial set of {num_frames} frames...")
    
    # Process initial frames
    for i, frame_idx in enumerate(frame_indices):
        # Set the position of the video to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame {frame_idx}")
            continue
        
        # Process the frame with SAM
        processed_frame, mask = apply_sam(frame, predictor)
        
        # Evaluate frame quality
        quality_score, metrics = evaluate_frame_quality(mask)
        
        frame_info = {
            "index": frame_idx,
            "processed_frame": processed_frame,
            "mask": mask,
            "quality_score": quality_score,
            "metrics": metrics
        }
        
        if quality_score >= quality_threshold:
            good_frames.append(frame_info)
            print(f"Frame {i+1}/{num_frames} (idx {frame_idx}) - Quality: {quality_score:.4f} - ACCEPTED")
        else:
            rejected_frames.append(frame_info)
            print(f"Frame {i+1}/{num_frames} (idx {frame_idx}) - Quality: {quality_score:.4f} - REJECTED")
    
    print(f"Initial processing: {len(good_frames)} good frames, {len(rejected_frames)} rejected frames")
    
    # Try to find replacement frames if needed
    if len(good_frames) < num_frames and len(rejected_frames) > 0:
        missing_frames = num_frames - len(good_frames)
        frames_to_try = min(missing_frames * 2, max_extra_frames)
        
        print(f"Looking for {missing_frames} replacement frames...")
        
        # Find gaps in the current frame sequence where we can insert additional frames
        existing_indices = sorted([f["index"] for f in good_frames])
        
        # Find the largest gaps
        gaps = []
        for i in range(len(existing_indices) - 1):
            gap_size = existing_indices[i+1] - existing_indices[i]
            if gap_size > 2:  # Minimum gap size to consider
                gaps.append((existing_indices[i], existing_indices[i+1], gap_size))
        
        # Sort gaps by size (largest first)
        gaps.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate new frames to try from the largest gaps
        additional_indices = []
        frames_added = 0
        
        for start, end, gap_size in gaps:
            # Number of frames to add in this gap
            n_frames = min(gap_size - 1, max(1, int(gap_size * missing_frames / total_frames)))
            
            for i in range(1, n_frames + 1):
                # Calculate position within the gap
                pos = start + (i * gap_size) // (n_frames + 1)
                additional_indices.append(pos)
                frames_added += 1
                
                if frames_added >= frames_to_try:
                    break
            
            if frames_added >= frames_to_try:
                break
        
        print(f"Trying {len(additional_indices)} additional frames from {len(gaps)} gaps")
        
        # Process additional frames
        for i, frame_idx in enumerate(additional_indices):
            # Set the position of the video to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read the frame
            ret, frame = cap.read()
            
            if not ret:
                print(f"Failed to read additional frame {frame_idx}")
                continue
            
            # Process the frame with SAM
            processed_frame, mask = apply_sam(frame, predictor)
            
            # Evaluate frame quality
            quality_score, metrics = evaluate_frame_quality(mask)
            
            frame_info = {
                "index": frame_idx,
                "processed_frame": processed_frame,
                "mask": mask,
                "quality_score": quality_score,
                "metrics": metrics
            }
            
            if quality_score >= quality_threshold:
                good_frames.append(frame_info)
                print(f"Additional frame {i+1}/{len(additional_indices)} (idx {frame_idx}) - Quality: {quality_score:.4f} - ACCEPTED")
                
                if len(good_frames) >= num_frames:
                    print(f"Reached target of {num_frames} good frames")
                    break
            else:
                print(f"Additional frame {i+1}/{len(additional_indices)} (idx {frame_idx}) - Quality: {quality_score:.4f} - REJECTED")
    
    # Sort good frames by their index to maintain chronological order
    good_frames.sort(key=lambda x: x["index"])
    
    # If we couldn't find enough good frames, use the best of the rejected ones
    if len(good_frames) < num_frames and rejected_frames:
        still_missing = num_frames - len(good_frames)
        print(f"Could only find {len(good_frames)} good frames. Using the {still_missing} best rejected frames.")
        
        # Sort rejected frames by quality score (best first)
        rejected_frames.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Add the best rejected frames
        good_frames.extend(rejected_frames[:still_missing])
        
        # Re-sort by index
        good_frames.sort(key=lambda x: x["index"])
    
    # Save the final set of frames
    print(f"Saving {len(good_frames)} frames...")
    
    for i, frame_info in enumerate(good_frames):
        # Get processed frame
        processed_frame = frame_info["processed_frame"]
        
        # Convert RGB to BGR for OpenCV
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Save the processed frame
        output_path = os.path.join(output_dir, f"img_{i+1}.jpg")
        cv2.imwrite(output_path, processed_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"Saved frame {i+1}/{len(good_frames)} (idx {frame_info['index']}) - Quality: {frame_info['quality_score']:.4f}")
    
    # Release the video
    cap.release()
    print(f"Completed processing {len(good_frames)} frames")
    
    return len(good_frames)

def preprocess_video(video_path, output_dir, num_frames=60):
    """
    Main preprocessing function to extract frames and remove backgrounds.
    
    Args:
        video_path (str): Path to the input video
        output_dir (str): Directory to save processed frames
        num_frames (int): Number of frames to extract
    """
    print(f"Preprocessing video: {video_path}")
    print(f"Extracting {num_frames} frames and applying SAM for background removal")
    
    processed_frames = extract_and_process_frames(video_path, output_dir, num_frames)
    
    print(f"Preprocessing completed successfully with {processed_frames} frames")
    return processed_frames

if __name__ == "__main__":
    # For testing when run directly
    video_path = "../data/NissanMurano/stable.mp4"
    output_dir = "../data/NissanMurano/colmap/images"
    preprocess_video(video_path, output_dir)