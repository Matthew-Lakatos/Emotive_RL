"""
features:
- Captures camera frames (uses DeepFace or OpenFace if installed; otherwise a simple stub).
- Records short audio and computes a spectral-entropy-based stress proxy.
- Fuses face valence + audio entropy into a single emotion scalar every `publish_interval` seconds.
- Publishes:
    - /emotion_scalar (std_msgs/Float32)    -- scalar emotional modulation
    - /agent/belief (std_msgs/Float32MultiArray) -- fused belief vector (example format)
- Logs JSON entries for each emission to logs/emotion_log.json for ethical auditing.
- Optionally loads a saved PyTorch PredictiveEmotion model and uses it to refine the scalar.
"""

import os
import json
import time
from datetime import datetime
from collections import deque

import numpy as np
import threading

# ROS imports
import rospy
from std_msgs.msg import Float32, Float32MultiArray

try:
    import cv2
except Exception:
    cv2 = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

try:
    import torch
    from predictive_emotion import PredictiveEmotion
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# Simple sliding window for smoothing
SMOOTH_WINDOW = 5


# -------------------------
# Utility: audio entropy
# -------------------------
def compute_audio_entropy(samples, samplerate):
    """
    Compute a simple spectral entropy as a stress proxy.
    - samples: 1D numpy array of audio samples
    - samplerate: sampling rate in Hz
    Returns: entropy (float, >=0). We scale later.
    """
    if samples is None or len(samples) == 0:
        return 0.0
    # Compute power spectrum
    fft = np.fft.rfft(samples)
    power = np.abs(fft) ** 2
    total = np.sum(power)
    if total <= 0:
        return 0.0
    p = power / total
    eps = 1e-12
    entropy = -np.sum(p * np.log(p + eps))
    return float(entropy)


def analyze_frame_valence(frame_bgr):
    """
    Analyze a BGR OpenCV frame and return a valence score in [-1, 1]
    - Uses DeepFace if available (returns "emotion" scores), else a heuristic stub.
    """
    if frame_bgr is None:
        return 0.0, None

    # Try deepface if installed
    if DEEPFACE_AVAILABLE:
        try:
            # DeepFace.analyze expects RGB in some versions; we'll pass frame and let it handle
            analysis = DeepFace.analyze(frame_bgr, actions=["emotion"], enforce_detection=False)
            # analysis["emotion"] is dict of emotion->score
            emotions = analysis.get("emotion", {})
            # compute valence: (positive - negative) normalized
            positive = emotions.get("happy", 0.0) + emotions.get("surprise", 0.0)
            negative = emotions.get("sad", 0.0) + emotions.get("angry", 0.0) + emotions.get("fear", 0.0) + emotions.get("disgust", 0.0)
            if (positive + negative) == 0:
                valence = 0.0
            else:
                valence = (positive - negative) / (positive + negative)
            # clamp to [-1, 1]
            valence = max(-1.0, min(1.0, valence))
            return float(valence), emotions
        except Exception as e:
            rospy.logwarn_throttle(30, f"DeepFace analysis failed: {e}")
            # fallback to stub below
    # Fallback heuristic: simple brightness-based proxy (NOT a real emotion model)
    try:
        # convert to grayscale mean intensity -> map to [-1,1]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if cv2 is not None else None
        mean_intensity = float(np.mean(gray)) if gray is not None else 127.5
        valence = (mean_intensity - 127.5) / 127.5  # roughly -1..1
        return float(max(-1.0, min(1.0, valence))), {"stub": True, "mean_intensity": mean_intensity}
    except Exception:
        return 0.0, {"stub": True}


# -------------------------
# The ROS Node
# -------------------------
class EmotionNode:
    def __init__(self,
                 model_path=None,
                 camera_index=0,
                 audio_samplerate=16000,
                 audio_duration=1.0,
                 publish_interval=2.0):
        """
        model_path: optional path to a saved PredictiveEmotion model (pytorch .pt)
        camera_index: camera device index for OpenCV
        audio_*: audio recording params
        publish_interval: seconds between publishes (paper: 2 seconds)
        """
        rospy.init_node("emotion_agent_node", anonymous=False)

        self.pub_scalar = rospy.Publisher("/emotion_scalar", Float32, queue_size=10)
        self.pub_belief = rospy.Publisher("/agent/belief", Float32MultiArray, queue_size=10)

        self.camera_index = camera_index
        self.audio_samplerate = audio_samplerate
        self.audio_duration = audio_duration
        self.publish_interval = publish_interval

        self.cap = None
        if cv2 is not None:
            try:
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    rospy.logwarn(f"Camera index {self.camera_index} not available; face inference disabled.")
                    self.cap = None
            except Exception as e:
                rospy.logwarn(f"OpenCV camera init failed: {e}")
                self.cap = None
        else:
            rospy.logwarn("OpenCV not available; face analysis disabled.")

        self.sd_available = sd is not None
        if not self.sd_available:
            rospy.logwarn("sounddevice not available; audio-based stress inference disabled.")

        # Load optional predictive emotion model (if provided)
        self.emotion_model = None
        if model_path and TORCH_AVAILABLE and os.path.exists(model_path):
            try:
                # assume the saved file contains state_dict for PredictiveEmotion
                model = PredictiveEmotion(input_dim=8, hidden_dim=32)  # input_dim depends on your feature design
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.eval()
                self.emotion_model = model
                rospy.loginfo("Loaded predictive emotion model from: " + model_path)
            except Exception as e:
                rospy.logwarn(f"Failed to load emotion model: {e}")

        # Smoothing buffers
        self.valence_history = deque(maxlen=SMOOTH_WINDOW)
        self.audio_entropy_history = deque(maxlen=SMOOTH_WINDOW)
        self.fused_history = deque(maxlen=SMOOTH_WINDOW)

        # Start worker thread to run sampling loop (so ROS spin doesn't block)
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

    def _capture_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _record_audio(self, duration):
        """
        Records `duration` seconds of audio and returns a numpy array of samples.
        If sounddevice isn't available, returns None.
        """
        if not self.sd_available:
            return None
        try:
            samples = sd.rec(int(duration * self.audio_samplerate), samplerate=self.audio_samplerate, channels=1, dtype="float32")
            sd.wait()
            samples = samples.squeeze()
            return samples
        except Exception as e:
            rospy.logwarn_throttle(30, f"Audio recording failed: {e}")
            return None

    def _fuse_signals(self, valence, audio_entropy):
        """
        Fuse valence ([-1,1]) and audio_entropy (>=0) into a single scalar in [-1,1]
        - Normalize audio entropy by recent typical range (this is heuristic; adjust per deployment)
        - Use weighted sum and tanh to bound
        """
        # Normalize audio entropy using running history
        if audio_entropy is None:
            a_norm = 0.0
        else:
            # running average based normalization
            hist = list(self.audio_entropy_history)
            if len(hist) >= 2:
                mean_hist = float(np.mean(hist))
                std_hist = float(np.std(hist)) + 1e-6
                a_norm = (audio_entropy - mean_hist) / std_hist
                # clamp to [-2, 2] then scale
                a_norm = max(-2.0, min(2.0, a_norm)) / 2.0  # now roughly [-1,1]
            else:
                # first samples: scale down
                a_norm = np.tanh(audio_entropy / (1.0 + audio_entropy))

        # weights: valence positive -> more positive, audio_entropy (stress) pushes negative
        w_valence = 0.7
        w_audio = -0.6  # negative weight because more entropy ~ more stress ~ negative valence

        fused_raw = w_valence * valence + w_audio * a_norm
        fused = float(np.tanh(fused_raw))  # final bounded scalar in (-1,1)

        return fused

    def _refine_with_emotion_model(self, fused_scalar):
        """
        If a predictive emotion model is present, optionally refine the fused scalar using it.
        This expects you design a mapping from fused scalar -> model input format.
        Here we demonstrate a simple way: create a dummy sequence where each timestep's 'feature'
        includes fused_scalar and zeros. Replace with your actual observation sequence.
        """
        if self.emotion_model is None:
            return fused_scalar

        try:
            # Create a dummy input sequence: batch=1, time=10, features=8 (example)
            seq_len = 10
            input_dim = 8
            x = torch.zeros((1, seq_len, input_dim), dtype=torch.float32)
            # place fused scalar in the first feature across time
            x[:, :, 0] = fused_scalar
            with torch.no_grad():
                e_hat = self.emotion_model(x).item()
            # combine e_hat and fused_scalar (example weighting)
            refined = 0.5 * fused_scalar + 0.5 * float(np.tanh(e_hat))
            return float(max(-1.0, min(1.0, refined)))
        except Exception as e:
            rospy.logwarn_throttle(30, f"Emotion model refine failed: {e}")
            return fused_scalar

    def _publish(self, scalar, belief_vector):
        # Publish scalar
        try:
            msg = Float32()
            msg.data = float(scalar)
            self.pub_scalar.publish(msg)
        except Exception as e:
            rospy.logerr_throttle(30, f"Failed to publish /emotion_scalar: {e}")

        # Publish belief vector as Float32MultiArray (example format: [agent_features..., emotion_scalar])
        try:
            arr = Float32MultiArray()
            arr.data = [float(x) for x in belief_vector]
            self.pub_belief.publish(arr)
        except Exception as e:
            rospy.logerr_throttle(30, f"Failed to publish /agent/belief: {e}")

    def _log_and_audit(self, entry):
        append_json_log(entry)

    def _run_loop(self):
        """
        Main loop: capture sensors, analyze, fuse, publish, log.
        Runs until node shutdown or stop event set.
        """
        rate = rospy.Rate(1.0 / max(1e-6, self.publish_interval))  # in Hz: 1 / interval
        rospy.loginfo("Emotion node loop started. Publishing every {:.2f}s".format(self.publish_interval))

        while not rospy.is_shutdown() and not self._stop_event.is_set():
            ts = datetime.utcnow().isoformat()

            # --- camera / face ---
            frame = self._capture_frame()
            valence, face_meta = analyze_frame_valence(frame) if frame is not None else (0.0, None)
            self.valence_history.append(valence)

            # --- audio ---
            audio_samples = self._record_audio(self.audio_duration)
            audio_entropy = compute_audio_entropy(audio_samples, self.audio_samplerate) if audio_samples is not None else None
            if audio_entropy is not None:
                self.audio_entropy_history.append(audio_entropy)

            # --- fuse ---
            fused = self._fuse_signals(valence, audio_entropy)
            # smooth with history
            smoothed = float(np.mean(list(self.fused_history) + [fused])) if len(self.fused_history) > 0 else fused
            self.fused_history.append(smoothed)

            # --- optional: refine via PredictiveEmotion model ---
            refined = self._refine_with_emotion_model(smoothed)

            # --- form belief vector for agent (example) ---
            # You should adapt this to your agent's expected belief vector shape.
            # Example belief vector: [recent_valence_avg, recent_audio_entropy_avg, refined_scalar]
            belief_vector = [
                float(np.mean(self.valence_history)) if len(self.valence_history) > 0 else 0.0,
                float(np.mean(self.audio_entropy_history)) if len(self.audio_entropy_history) > 0 else 0.0,
                refined
            ]

            # --- publish to ROS topics ---
            self._publish(refined, belief_vector)

            # --- JSON log entry for auditing (consent checks should run outside this node) ---
            log_entry = {
                "timestamp_utc": ts,
                "valence": valence,
                "face_meta": face_meta,
                "audio_entropy": audio_entropy,
                "fused_raw": fused,
                "smoothed": smoothed,
                "refined": refined,
                "belief_vector": belief_vector
            }
            self._log_and_audit(log_entry)

            # Sleep until next publish
            rate.sleep()

        # cleanup on shutdown
        rospy.loginfo("Emotion node shutting down; releasing camera.")
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)


# -------------------------
# CLI entrypoint
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="ROS Emotion Node (Camera + Audio -> emotion scalar)")
    parser.add_argument("--model", type=str, default=None, help="Path to saved PredictiveEmotion model (.pt)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (OpenCV)")
    parser.add_argument("--audio_rate", type=int, default=16000, help="Audio sampling rate for recording")
    parser.add_argument("--audio_dur", type=float, default=1.0, help="Audio recording duration (seconds)")
    parser.add_argument("--interval", type=float, default=2.0, help="Publish interval (seconds) - paper uses 2s")
    args = parser.parse_args(rospy.myargv()[1:])

    node = EmotionNode(model_path=args.model,
                       camera_index=args.camera,
                       audio_samplerate=args.audio_rate,
                       audio_duration=args.audio_dur,
                       publish_interval=args.interval)

    rospy.loginfo("Emotion node running. Ctrl+C to stop.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user.")
    finally:
        node.stop()


if __name__ == "__main__":
    main()
