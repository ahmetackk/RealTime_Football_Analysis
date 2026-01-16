"""
UNSUPERVISED ANALYSIS SPEED PROFILES
Kullanıcı GUI'den hız profilini seçer, script'ler bu ayarları okur.
"""

SPEED_PROFILES = {
    "fast": {
        "name": "Fast",
        "description": "Quick demo (~2-3 minutes)",
        "unsupervised_samples": 80,       # ✅ 50 -> 80 (more stable)
        "clip_max_per_class": 3,
        "new_action_frame_limit": 1000,   # ✅ 750 -> 1000
        "new_action_stride": 100,
        "new_action_videos_per_cluster": 2
    },
    "medium": {
        "name": "Medium",
        "description": "Balanced results (~5-7 minutes)",
        "unsupervised_samples": 150,      # ✅ 100 -> 150
        "clip_max_per_class": 5,
        "new_action_frame_limit": 2000,   # ✅ 1500 -> 2000
        "new_action_stride": 75,
        "new_action_videos_per_cluster": 3
    },
    "slow": {
        "name": "Slow",
        "description": "Comprehensive analysis (~15-20 minutes)",
        "unsupervised_samples": 300,      # ✅ 200 -> 300
        "clip_max_per_class": 10,
        "new_action_frame_limit": 4000,   # ✅ 3000 -> 4000
        "new_action_stride": 50,
        "new_action_videos_per_cluster": 5
    }
}

def get_profile(speed_name):
    """
    Speed profile'ı al
    
    Args:
        speed_name: "fast", "medium", veya "slow"
    
    Returns:
        dict: Profile ayarları
    """
    return SPEED_PROFILES.get(speed_name, SPEED_PROFILES["medium"])

def save_current_profile(speed_name):
    """
    Seçilen profili dosyaya kaydet (script'ler okuyacak)
    """
    import json
    profile = get_profile(speed_name)
    
    with open("unsupervised_speed_config.json", "w") as f:
        json.dump(profile, f, indent=2)
    
    return profile

def load_current_profile():
    """
    Kaydedilmiş profili oku
    """
    import json
    import os
    
    if os.path.exists("unsupervised_speed_config.json"):
        with open("unsupervised_speed_config.json", "r") as f:
            return json.load(f)
    
    # Default: medium
    return SPEED_PROFILES["medium"]