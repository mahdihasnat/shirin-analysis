import os

def generate_gallery(frames_dir, output_file="index.html"):
    """
    Generates an HTML gallery for images in the frames directory.
    
    Args:
        frames_dir (str): Path to the directory containing images.
        output_file (str): Path to the output HTML file.
    """
    # Use absolute paths for checking file existence but relative for HTML
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve frames_dir relative to the script location
    if not os.path.isabs(frames_dir):
        abs_frames_dir = os.path.normpath(os.path.join(script_dir, frames_dir))
    else:
        abs_frames_dir = frames_dir

    if not os.path.exists(abs_frames_dir):
        print(f"Error: Frames directory not found: {abs_frames_dir}")
        return

    images = sorted([f for f in os.listdir(abs_frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extracted Frames Gallery</title>
    <style>
        body {{
            font-family: sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            padding: 10px;
        }}
        .gallery-item {{
            background-color: white;
            padding: 5px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .gallery-item img {{
            width: 100%;
            height: auto;
            display: block;
            border-radius: 3px;
        }}
        .gallery-item p {{
            margin: 5px 0 0;
            font-size: 0.8em;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>Extracted Frames Gallery</h1>
    <div class="gallery">
"""
    
    for img in images:
        # Construct path relative to the HTML file location (which is script_dir)
        # We assume frames_dir path passed in is already relative or we make it relative
        
        # Calculate relative path from script_dir to image file
        img_abs_path = os.path.join(abs_frames_dir, img)
        rel_path = os.path.relpath(img_abs_path, script_dir)
        
        html_content += f"""
        <div class="gallery-item">
            <img src="{rel_path}" alt="{img}" loading="lazy">
            <p>{img}</p>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    output_path = os.path.join(script_dir, output_file)
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"Gallery generated at: {output_path}")

if __name__ == "__main__":
    # Frames are in ../frame_extractor/output relative to this script
    frames_directory = "../frame_extractor/output"
    generate_gallery(frames_directory)
