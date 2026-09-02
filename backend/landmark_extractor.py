import numpy as np

def normalize_landmarks(landmarks):
    """
    Normalizes 21 3D hand landmarks:
    - Centers coordinates relative to the wrist (landmark 0)
    - Scales features by maximum distance from wrist (scale-invariant)
    - Extracts coordinate vector + key distances & joint angles
    landmarks: list of dicts [{'x': x, 'y': y, 'z': z}, ...] or list of [x, y, z]
    Returns 1D numpy array of normalized feature vector.
    """
    if landmarks is None or len(landmarks) != 21:
        return None

    pts = []
    for lm in landmarks:
        if isinstance(lm, dict):
            pts.append([lm['x'], lm['y'], lm['z']])
        elif isinstance(lm, (list, tuple, np.ndarray)):
            pts.append([lm[0], lm[1], lm[2] if len(lm) > 2 else 0.0])
        else:
            pts.append([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])
    
    pts = np.array(pts, dtype=np.float32)
    if not np.isfinite(pts).all():
        return None

    # 1. Translate wrist (point 0) to origin
    wrist = pts[0]
    centered = pts - wrist

    # 2. Scale normalization
    distances = np.linalg.norm(centered, axis=1)
    max_dist = np.max(distances)
    if max_dist > 1e-6:
        scaled = centered / max_dist
    else:
        scaled = centered

    # 3. Flat coordinate features (63 elements)
    coord_features = scaled.flatten()

    # 4. Auxiliary geometric features (distances between fingertips and wrist/base)
    # Fingertips: 4 (Thumb), 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
    tips = [4, 8, 12, 16, 20]
    bases = [2, 5, 9, 13, 17]

    tip_distances = [np.linalg.norm(scaled[t]) for t in tips]
    
    # Inter-finger distances (thumb-index, index-middle, etc.)
    inter_finger = [
        np.linalg.norm(scaled[8] - scaled[4]),   # Index-Thumb
        np.linalg.norm(scaled[12] - scaled[8]),  # Middle-Index
        np.linalg.norm(scaled[16] - scaled[12]), # Ring-Middle
        np.linalg.norm(scaled[20] - scaled[16]), # Pinky-Ring
        np.linalg.norm(scaled[20] - scaled[4]),  # Pinky-Thumb
    ]

    # Finger curl status (distance from tip to wrist vs base to wrist ratio)
    curl_ratios = []
    for t, b in zip(tips, bases):
        dist_tip = np.linalg.norm(scaled[t])
        dist_base = np.linalg.norm(scaled[b])
        curl_ratios.append(dist_tip / (dist_base + 1e-6))

    # Combine into single feature vector
    feature_vector = np.concatenate([
        coord_features,
        tip_distances,
        inter_finger,
        curl_ratios
    ])

    return feature_vector
