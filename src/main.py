import os
from vid_stab import stabilize_video_ffmpeg

def ensure_output_directory(output_path):
    """
    Ensure the output directory exists.
    
    Args:
        output_path (str): Path to the output file
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    """
    Entrypoint of all code.
    God Bless. Never Crash.
    """
    # these will be set up to be dynamic later
    input_file = "../data/NissanMurano/unstable.mp4"
    output_file = "../data/NissanMurano/stable_max_smoothing.mp4"
    
    # make sure the output directory exists
    ensure_output_directory(output_file)
    
    # Stabilize with max smooth quality parameters
    success = stabilize_video_ffmpeg(input_file, output_file)
    
    if success:
        print("Video stabilization completed successfully")
    else:
        print("Video stabilization failed")

if __name__ == "__main__":
    main()