import os
from vid_stab import stabilize_video_ffmpeg
from preprocess import preprocess_video

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
    output_file = "../data/NissanMurano/stable.mp4"
    frames_output_dir = "../data/NissanMurano/colmap/images"
    
    # make sure the output directory exists
    ensure_output_directory(output_file)
    
    # Stabilize with max smooth quality parameters
    success = stabilize_video_ffmpeg(input_file, output_file)
    if not success:
        print("Video stabilization failed")
        return 1
    
    print("Video stabilization completed successfully")
    
    # Preprocess the stabilized video: extract frames and remove backgrounds
    preprocess_success = preprocess_video(output_file, frames_output_dir, num_frames=60)
    if not preprocess_success:
        print("Preprocessing failed")
        return 1

if __name__ == "__main__":
    main()