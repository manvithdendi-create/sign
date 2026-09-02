import numpy as np
from backend.landmark_extractor import normalize_landmarks

# Base 3D landmark definitions for core sign gestures (21 points [x, y, z])
# Standard hand skeleton keypoints:
# 0: Wrist, 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky

def generate_base_templates():
    templates = {}

    # Helper function to generate realistic hand shape coordinates
    def make_hand(thumb='folded', index='folded', middle='folded', ring='folded', pinky='folded',
                  spread=True, pointing='up', crossed=False, c_shape=False, o_shape=False):
        pts = np.zeros((21, 3), dtype=np.float32)
        # Wrist at [0.0, 0.5, 0.0]
        pts[0] = [0.0, 0.5, 0.0]
        
        # Base palm joints (MCPs)
        pts[1]  = [-0.1, 0.4, 0.0]   # Thumb CMC
        pts[2]  = [-0.18, 0.3, 0.0]  # Thumb MCP
        pts[5]  = [-0.15, 0.2, 0.0]  # Index MCP
        pts[9]  = [-0.05, 0.18, 0.0] # Middle MCP
        pts[13] = [0.05, 0.2, 0.0]   # Ring MCP
        pts[17] = [0.15, 0.25, 0.0]  # Pinky MCP

        if c_shape:
            # Curved C-shape profile
            pts[3]  = [-0.22, 0.2, -0.1]
            pts[4]  = [-0.25, 0.0, -0.2]  # Thumb Tip
            
            pts[6]  = [-0.15, 0.0, -0.1]
            pts[7]  = [-0.15, -0.1, -0.15]
            pts[8]  = [-0.15, -0.2, -0.2] # Index Tip
            
            pts[10] = [-0.05, 0.0, -0.1]
            pts[11] = [-0.05, -0.12, -0.15]
            pts[12] = [-0.05, -0.22, -0.2] # Middle Tip
            
            pts[14] = [0.05, 0.0, -0.1]
            pts[15] = [0.05, -0.1, -0.15]
            pts[16] = [0.05, -0.2, -0.2]  # Ring Tip
            
            pts[18] = [0.15, 0.0, -0.1]
            pts[19] = [0.15, -0.1, -0.15]
            pts[20] = [0.15, -0.15, -0.2] # Pinky Tip
            return pts

        if o_shape:
            # Circle O-shape profile (all tips meet thumb)
            pts[3]  = [-0.1, 0.15, -0.05]
            pts[4]  = [-0.05, 0.05, -0.1] # Thumb Tip
            
            pts[6]  = [-0.12, 0.12, -0.05]
            pts[7]  = [-0.08, 0.08, -0.08]
            pts[8]  = [-0.05, 0.05, -0.1] # Index Tip
            
            pts[10] = [-0.05, 0.12, -0.05]
            pts[11] = [-0.05, 0.08, -0.08]
            pts[12] = [-0.04, 0.05, -0.1] # Middle Tip
            
            pts[14] = [0.03, 0.12, -0.05]
            pts[15] = [0.0, 0.08, -0.08]
            pts[16] = [-0.04, 0.05, -0.1] # Ring Tip
            
            pts[18] = [0.1, 0.15, -0.05]
            pts[19] = [0.05, 0.08, -0.08]
            pts[20] = [-0.03, 0.05, -0.1] # Pinky Tip
            return pts

        # Base finger logic
        # Directions for extended fingers
        if pointing == 'side':
            ext_dir_base = np.array([-1.0, 0.0, 0.0])
            spread_axis = np.array([0.0, -1.0, 0.0])
        elif pointing == 'down':
            ext_dir_base = np.array([0.0, 1.0, 0.0])
            spread_axis = np.array([-1.0, 0.0, 0.0])
        else: # up
            ext_dir_base = np.array([0.0, -1.0, 0.0])
            spread_axis = np.array([1.0, 0.0, 0.0])

        # Extended direction offsets per finger
        if spread:
            idx_offset = -0.22 * spread_axis
            mid_offset = 0.0 * spread_axis
            rng_offset = 0.22 * spread_axis
            pky_offset = 0.44 * spread_axis
        else: # together
            idx_offset = -0.04 * spread_axis
            mid_offset = 0.0 * spread_axis
            rng_offset = 0.04 * spread_axis
            pky_offset = 0.08 * spread_axis

        # If crossed (for R)
        if crossed:
            idx_offset, mid_offset = mid_offset, idx_offset
            idx_z = 0.05
            mid_z = -0.05
        else:
            idx_z = 0.0
            mid_z = 0.0

        # Thumb logic
        if thumb == 'ext':
            pts[3] = [-0.28, 0.22, 0.0]
            pts[4] = [-0.38, 0.12, 0.0]
        elif thumb == 'cross':
            pts[3] = [-0.05, 0.26, 0.05]
            pts[4] = [0.02, 0.28, 0.05]
        elif thumb == 'up':
            pts[3] = [-0.15, 0.24, -0.05]
            pts[4] = [-0.12, 0.12, -0.08]
        else: # folded
            pts[3] = [-0.12, 0.28, 0.02]
            pts[4] = [-0.08, 0.25, 0.02]

        # Helper to construct coordinates for index, middle, ring, pinky
        def set_finger_coords(start_idx, mcp_pt, state, ext_dir, ext_z):
            if state == 'ext':
                pts[start_idx]   = mcp_pt + ext_dir * 0.15 + [0, 0, ext_z]
                pts[start_idx+1] = mcp_pt + ext_dir * 0.30 + [0, 0, ext_z]
                pts[start_idx+2] = mcp_pt + ext_dir * 0.45 + [0, 0, ext_z]
            elif state == 'hooked':
                pts[start_idx]   = mcp_pt + ext_dir * 0.15
                pts[start_idx+1] = mcp_pt + ext_dir * 0.25 + [-ext_dir[1]*0.1, ext_dir[0]*0.1, 0.1]
                pts[start_idx+2] = mcp_pt + ext_dir * 0.18 + [-ext_dir[1]*0.18, ext_dir[0]*0.18, 0.15]
            else: # folded
                pts[start_idx]   = mcp_pt + [0.03, 0.05, 0.05]
                pts[start_idx+1] = mcp_pt + [0.05, 0.08, 0.05]
                pts[start_idx+2] = mcp_pt + [0.04, 0.04, 0.02]

        # Index (points 5-8)
        set_finger_coords(6, pts[5], index, ext_dir_base + idx_offset, idx_z)
        # Middle (points 9-12)
        set_finger_coords(10, pts[9], middle, ext_dir_base + mid_offset, mid_z)
        # Ring (points 13-16)
        set_finger_coords(14, pts[13], ring, ext_dir_base + rng_offset, 0.0)
        # Pinky (points 17-20)
        set_finger_coords(18, pts[17], pinky, ext_dir_base + pky_offset, 0.0)

        return pts

    # Define common ASL Signs & Gestures
    # Letters
    templates['A'] = make_hand(thumb='ext', index='folded', middle='folded', ring='folded', pinky='folded')
    templates['B'] = make_hand(thumb='cross', index='ext', middle='ext', ring='ext', pinky='ext', spread=False)
    templates['C'] = make_hand(c_shape=True)
    templates['D'] = make_hand(thumb='cross', index='ext', middle='folded', ring='folded', pinky='folded')
    templates['E'] = make_hand(o_shape=True)
    templates['F'] = make_hand(thumb='cross', index='folded', middle='ext', ring='ext', pinky='ext', spread=True)
    templates['G'] = make_hand(thumb='ext', index='ext', middle='folded', ring='folded', pinky='folded', pointing='side')
    templates['H'] = make_hand(thumb='folded', index='ext', middle='ext', ring='folded', pinky='folded', pointing='side', spread=False)
    templates['I'] = make_hand(thumb='folded', index='folded', middle='folded', ring='folded', pinky='ext')
    templates['J'] = make_hand(thumb='ext', index='folded', middle='folded', ring='folded', pinky='ext')
    templates['K'] = make_hand(thumb='up', index='ext', middle='ext', ring='folded', pinky='folded', spread=True)
    templates['L'] = make_hand(thumb='ext', index='ext', middle='folded', ring='folded', pinky='folded')
    templates['M'] = make_hand(thumb='cross', index='folded', middle='folded', ring='folded', pinky='folded')
    templates['N'] = make_hand(thumb='folded', index='folded', middle='folded', ring='folded', pinky='folded')
    templates['O'] = make_hand(o_shape=True)
    templates['P'] = make_hand(thumb='ext', index='ext', middle='ext', ring='folded', pinky='folded', pointing='down')
    templates['Q'] = make_hand(thumb='ext', index='ext', middle='folded', ring='folded', pinky='folded', pointing='down')
    templates['R'] = make_hand(thumb='folded', index='ext', middle='ext', ring='folded', pinky='folded', spread=False, crossed=True)
    templates['S'] = make_hand(thumb='cross', index='folded', middle='folded', ring='folded', pinky='folded')
    templates['T'] = make_hand(thumb='up', index='folded', middle='folded', ring='folded', pinky='folded')
    templates['U'] = make_hand(thumb='folded', index='ext', middle='ext', ring='folded', pinky='folded', spread=False)
    templates['V'] = make_hand(thumb='folded', index='ext', middle='ext', ring='folded', pinky='folded', spread=True)
    templates['W'] = make_hand(thumb='folded', index='ext', middle='ext', ring='ext', pinky='folded', spread=True)
    templates['X'] = make_hand(thumb='folded', index='hooked', middle='folded', ring='folded', pinky='folded')
    templates['Y'] = make_hand(thumb='ext', index='folded', middle='folded', ring='folded', pinky='ext')
    templates['Z'] = make_hand(thumb='folded', index='ext', middle='folded', ring='folded', pinky='folded')

    # Numbers
    templates['0'] = make_hand(o_shape=True)
    templates['1'] = make_hand(thumb='folded', index='ext', middle='folded', ring='folded', pinky='folded')
    templates['2'] = make_hand(thumb='folded', index='ext', middle='ext', ring='folded', pinky='folded', spread=False)
    templates['3'] = make_hand(thumb='ext', index='ext', middle='ext', ring='folded', pinky='folded', spread=True)
    templates['4'] = make_hand(thumb='folded', index='ext', middle='ext', ring='ext', pinky='ext', spread=True)
    templates['5'] = make_hand(thumb='ext', index='ext', middle='ext', ring='ext', pinky='ext', spread=True)
    templates['6'] = make_hand(thumb='folded', index='ext', middle='ext', ring='ext', pinky='folded')
    templates['7'] = make_hand(thumb='folded', index='ext', middle='ext', ring='folded', pinky='ext')
    templates['8'] = make_hand(thumb='folded', index='ext', middle='folded', ring='ext', pinky='ext')
    templates['9'] = make_hand(thumb='folded', index='folded', middle='ext', ring='ext', pinky='ext')

    # Words & Expressions
    templates['HELLO'] = make_hand(thumb='ext', index='ext', middle='ext', ring='ext', pinky='ext', spread=False)
    templates['PEACE'] = make_hand(thumb='folded', index='ext', middle='ext', ring='folded', pinky='folded', spread=True)
    templates['I_LOVE_YOU'] = make_hand(thumb='ext', index='ext', middle='folded', ring='folded', pinky='ext')
    templates['LIKE'] = make_hand(thumb='ext', index='folded', middle='folded', ring='folded', pinky='folded')
    templates['OK'] = make_hand(thumb='cross', index='folded', middle='ext', ring='ext', pinky='ext', spread=False)
    templates['NO'] = make_hand(thumb='ext', index='ext', middle='ext', ring='folded', pinky='folded', spread=False)
    templates['YES'] = make_hand(thumb='cross', index='folded', middle='folded', ring='folded', pinky='folded')

    return templates

def generate_synthetic_dataset(num_samples_per_class=120):
    """
    Generates synthetic landmark dataset with rotational, scale, and Gaussian noise augmentation
    to pre-seed the ML classifier.
    """
    templates = generate_base_templates()
    X = []
    y = []

    for label, base_pts in templates.items():
        for _ in range(num_samples_per_class):
            # Augment with random rotation around z-axis (-15 deg to +15 deg)
            angle = np.random.uniform(-0.25, 0.25)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            
            aug_pts = np.dot(base_pts, rot_matrix)
            
            # Augment with random scale (0.85 to 1.15)
            scale = np.random.uniform(0.85, 1.15)
            aug_pts = aug_pts * scale
            
            # Augment with Gaussian noise
            noise = np.random.normal(0, 0.012, size=aug_pts.shape)
            aug_pts += noise

            features = normalize_landmarks(aug_pts)
            if features is not None:
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)
