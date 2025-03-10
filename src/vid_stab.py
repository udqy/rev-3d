# FFmpeg Video Stabilization Implementation - Max Smooth Quality

import os
import subprocess
import tempfile

def stabilize_video_ffmpeg(input_path, output_path):
    """
    Stabilize video using FFmpeg's vidstab filter with max smooth quality configuration.
    
    Args:
        input_path (str): Path to input unstable video
        output_path (str): Path to save stabilized video
    
    Returns:
        bool: True if stabilization was successful, False otherwise
    """
    print(f"Stabilizing {input_path} to {output_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        return False
    
    # Create a temporary file for the transform data
    with tempfile.NamedTemporaryFile(suffix='.trf', delete=False) as tmp_file:
        transform_file = tmp_file.name
    
    try:
        # First pass: analyze video and generate transform data
        analyze_cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f'vidstabdetect=shakiness=10:accuracy=15:stepsize=4:mincontrast=0.3:result={transform_file}',
            '-f', 'null', '-'
        ]
        
        print("Step 1: Analyzing video motion...")
        subprocess.run(analyze_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Second pass: apply stabilization using the transform data
        stabilize_cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f'vidstabtransform=input={transform_file}:zoom=1.12:smoothing=50:optalgo=gauss:maxshift=-1:interpol=bicubic',
            '-c:v', 'libx264',  # Use x264 codec
            '-preset', 'slow',  # Slower preset for better quality
            '-tune', 'film',    # Tune for film content
            '-crf', '15',       # Higher quality (lower value)
            '-c:a', 'copy',     # Copy audio stream
            output_path
        ]
        
        print("Step 2: Applying stabilization...")
        subprocess.run(stabilize_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"Stabilization completed. Output saved to {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg processing: {e}")
        print(f"Error output: {e.stderr.decode() if e.stderr else 'None'}")
        return False
    finally:
        # Clean up the temporary transform file
        if os.path.exists(transform_file):
            os.remove(transform_file)
