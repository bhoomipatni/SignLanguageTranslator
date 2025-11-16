import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
import sys
from collections import deque

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import config
Config = config.Config

# -----------------------
# Helper functions
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def normalize_frame(hand_frames):
    """
    Normalize a single-frame keypoint vector (shape 126,)
    - If both hands zero, returns zeros.
    - Centers coordinates and scales by max spread (so scale invariant).
    """
    pts = hand_frames.reshape(-1, 3)  # (42, 3)
    # only use x,y for normalization scale/center (z kept but scaled same)
    xy = pts[:, :2]
    # find indices that are non-zero
    valid = np.any(xy != 0.0, axis=1)
    if not np.any(valid):
        return hand_frames  # all zeros

    valid_xy = xy[valid]
    # center at mean of valid points
    center = valid_xy.mean(axis=0)
    centered = xy - center

    # scale by max distance to origin among valid points (avoid div0)
    dists = np.linalg.norm(centered, axis=1)
    maxd = dists.max()
    if maxd < 1e-6:
        maxd = 1.0

    scaled_xy = centered / maxd
    pts[:, :2] = scaled_xy
    # scale z by same factor (optional)
    pts[:, 2] = pts[:, 2] / maxd
    return pts.flatten()

def smooth_sequence(seq, smoothing_window=3):
    """
    Apply a simple moving average along time axis.
    seq: np.array (T, D)
    smoothing_window: odd integer
    """
    if smoothing_window <= 1 or seq.shape[0] < smoothing_window:
        return seq
    k = smoothing_window
    pad = k // 2
    padded = np.pad(seq, ((pad, pad), (0,0)), mode='edge')
    smoothed = np.array([padded[i:i+k].mean(axis=0) for i in range(seq.shape[0])])
    return smoothed

# -----------------------
# Keypoint extractor class
# -----------------------
class KeypointExtractor:
    def __init__(self,
                 min_detection_confidence=0.4,
                 min_tracking_confidence=0.4,
                 smoothing_window=3):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.smoothing_window = smoothing_window

    def extract_from_video(self, video_path, max_frames=300, sample_rate=1, min_detected_frames=3):
        """
        Processes video and returns normalized, optionally-smoothed sequence of shape (T,126)
        or (None, error_string) if failed / low quality.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, "Could not open video"

        keypoints_list = []
        total_frames = 0
        detected_frames = 0
        frame_idx = 0

        while cap.isOpened() and total_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            total_frames += 1
            # convert and process
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # create per-frame vector with consistent left/right ordering
            frame_kp = self._frame_keypoints_from_results(results)
            if np.any(frame_kp != 0):
                detected_frames += 1
            keypoints_list.append(frame_kp)

        cap.release()

        if total_frames == 0:
            return None, "No frames in video"

        detection_rate = detected_frames / total_frames
        if detected_frames < min_detected_frames or detection_rate < 0.05:
            return None, f"Low hand detection ({detected_frames}/{total_frames} frames, {detection_rate:.1%})"

        seq = np.vstack(keypoints_list).astype(np.float32)  # (T,126)

        # Normalize per-frame
        seq_normed = np.array([normalize_frame(f) for f in seq], dtype=np.float32)

        # Optional temporal smoothing to reduce jitter
        seq_smoothed = smooth_sequence(seq_normed, smoothing_window=self.smoothing_window)

        return seq_smoothed, None

    def _frame_keypoints_from_results(self, results):
        """
        Builds a 126-dim vector for the frame:
        layout: [hand0(21*(x,y,z)), hand1(21*(x,y,z))], where hand0=LEFT, hand1=RIGHT.
        If a hand is missing, its 63 values are zeros.
        Hand ordering uses MediaPipe handedness information when available.
        """
        vector = np.zeros(126, dtype=np.float32)
        if not results or not results.multi_hand_landmarks:
            return vector

        # Default: fill hands based on handedness mapping if possible
        # multi_hand_landmarks and multi_handedness align by index
        landmarks_list = results.multi_hand_landmarks
        handedness_list = []
        if hasattr(results, "multi_handedness") and results.multi_handedness:
            # classification label can be 'Left' or 'Right'
            for h in results.multi_handedness:
                label = h.classification[0].label  # 'Left' or 'Right'
                handedness_list.append(label.lower())

        # Map landmarks to left/right indices
        for i, hand_landmarks in enumerate(landmarks_list):
            # default target index (0=left,1=right)
            target_idx = None
            if i < len(handedness_list):
                hd = handedness_list[i]
                if hd == 'left':
                    target_idx = 0
                elif hd == 'right':
                    target_idx = 1
            # If no handedness info, use index order
            if target_idx is None:
                target_idx = i if i < 2 else 1

            # write 21 landmarks * 3
            base = target_idx * 63
            for j, lm in enumerate(hand_landmarks.landmark):
                vector[base + j*3] = lm.x
                vector[base + j*3 + 1] = lm.y
                vector[base + j*3 + 2] = lm.z

        return vector

# -----------------------
# Main dataset processing
# -----------------------
def process_wlasl_dataset():
    """Process WLASL videos organized by word folders"""
    extractor = KeypointExtractor()
    
    # Create output directory
    output_dir = Path(Config.PROCESSED_DATA_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===========================================================
    # AUTO-CLEAR OLD KEYPOINTS EACH RUN
    # ===========================================================
    print(f"Clearing old keypoints in: {output_dir}")
    for file in output_dir.iterdir():
        try:
            if file.is_file():
                file.unlink()            # Delete file
            elif file.is_dir():
                import shutil
                shutil.rmtree(file)      # Delete folder
        except Exception as e:
            print(f"Could not delete {file}: {e}")
    print("✔ Cleared old processed data.\n")
    # ===========================================================
    
    # Look for videos directory
    videos_base_dir = Path(Config.RAW_DATA_PATH)
    
    if not videos_base_dir.exists():
        print(f"Videos directory not found at {videos_base_dir}")
        return


    word_folders = [d for d in videos_base_dir.iterdir() if d.is_dir()]
    if len(word_folders) == 0:
        print(f"No word folders found in {videos_base_dir}")
        return

    print(f"Found {len(word_folders)} word folders\n")

    processed_data = []
    labels_map = {}
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    failed_videos = []

    for word_idx, word_folder in enumerate(sorted(word_folders)):
        word = word_folder.name
        labels_map[word] = word_idx

        # collect video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(sorted(word_folder.glob(f'*{ext}')))

        if len(video_files) == 0:
            continue

        print(f"[{word_idx + 1}/{len(word_folders)}] '{word}' - {len(video_files)} videos")

        successful = 0
        for vid_num, video_path in enumerate(video_files):
            video_id = f"{word}_{vid_num:03d}"

            print(f"  [{vid_num+1}/{len(video_files)}] {video_path.name[:60]:60}", end=" ", flush=True)
            try:
                keypoints, error = extractor.extract_from_video(
                    video_path,
                    max_frames=Config.MAX_FRAMES if hasattr(Config, "MAX_FRAMES") else 300,
                    sample_rate=Config.SAMPLE_RATE if hasattr(Config, "SAMPLE_RATE") else 2,
                    min_detected_frames=Config.MIN_DETECTED_FRAMES if hasattr(Config, "MIN_DETECTED_FRAMES") else 3
                )

                if keypoints is not None:
                    out_file = output_dir / f"{video_id}.npy"
                    np.save(out_file, keypoints)

                    processed_data.append({
                        'video_id': video_id,
                        'word': word,
                        'label': word_idx,
                        'keypoints_file': str(out_file),
                        'num_frames': int(keypoints.shape[0]),
                        'source_video': str(video_path)
                    })
                    successful += 1
                    print(f"✓ ({keypoints.shape[0]}f)")
                else:
                    print(f"✗ {error}")
                    failed_videos.append({'word': word, 'file': video_path.name, 'reason': error})

            except Exception as e:
                print(f"✗ {str(e)}")
                failed_videos.append({'word': word, 'file': video_path.name, 'reason': str(e)})

        print(f"  Success: {successful}/{len(video_files)}\n")

    # Save labels map and index
    labels_path = Path(Config.LABELS_PATH)
    ensure_dir(labels_path.parent)
    with open(labels_path, 'w') as f:
        json.dump(labels_map, f, indent=2)

    with open(output_dir / "index.json", 'w') as f:
        json.dump(processed_data, f, indent=2)

    # Save failed list if any
    if failed_videos:
        with open(output_dir / "failed_videos.json", 'w') as f:
            json.dump(failed_videos, f, indent=2)

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Processed: {len(processed_data)} videos")
    print(f"Failed: {len(failed_videos)} videos")
    print(f"Words: {len(labels_map)}")

    word_counts = {}
    for entry in processed_data:
        word_counts[entry['word']] = word_counts.get(entry['word'], 0) + 1

    print("\nSamples per word:")
    for w in sorted(word_counts.keys()):
        print(f"  {w}: {word_counts[w]}")

    if failed_videos:
        print(f"\nFailed videos written to {output_dir / 'failed_videos.json'}")
    print("="*60)

if __name__ == "__main__":
    process_wlasl_dataset()
