This component creates frames from the movie.

Location of the movie is input/Shirin.Abbas.Kiarostami.2008.DVDRip.XViD.avi

All the generated frames will be stored in output/

## Setup

### Prerequisites (Ubuntu/Debian)
If you encounter an error creating the virtual environment, you may need to install the `python3-venv` package:
```bash
sudo apt install python3.12-venv
```

### Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script to extract frames:
```bash
python extract_frames.py
```