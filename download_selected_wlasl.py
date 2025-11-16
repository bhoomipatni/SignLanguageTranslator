import json
import os
import yt_dlp

# ===========================================================
# EDIT THIS LIST: which glosses (words) you want to download
# ===========================================================
TARGET_GLOSSES = ["HELLO", "PLEASE", "THANK YOU"]

# ===========================================================
# Path to JSON
# ===========================================================
JSON_PATH = "WLASL_v0.3.json"

# ===========================================================
# Output base folder
# ===========================================================
OUTPUT_DIR = "selected_wlasl_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================
# Load dataset
# ===========================================================
print(f"Loading dataset JSON: {JSON_PATH}")
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# ===========================================================
# Download loop
# ===========================================================
for entry in data:
    gloss = entry["gloss"].upper()

    if gloss not in [g.upper() for g in TARGET_GLOSSES]:
        continue

    # Create folder for this specific word
    word_folder = os.path.join(OUTPUT_DIR, gloss)
    os.makedirs(word_folder, exist_ok=True)

    print(f"\n==============================")
    print(f"   WORD: {gloss}")
    print(f"   Saving videos to: {word_folder}")
    print(f"==============================")

    # Set yt-dlp options *inside* loop so it resets path each time
    ydl_opts = {
        "format": "best",
        "quiet": False,
        "ignoreerrors": True,
        # FORCE videos into the word folder
        "outtmpl": os.path.join(word_folder, "%(id)s.%(ext)s")
    }

    downloader = yt_dlp.YoutubeDL(ydl_opts)

    # Loop through all instances for this gloss
    for inst in entry["instances"]:
        url = inst.get("url")
        vid = inst.get("video_id")

        if not url:
            print(f"Skipping {vid}: No URL available.")
            continue

        print(f"→ Downloading video_id {vid}")

        try:
            downloader.download([url])
        except Exception as e:
            print(f"FAILED for {vid}: {e}")

print("\nDONE! Check your folders inside:", OUTPUT_DIR)
