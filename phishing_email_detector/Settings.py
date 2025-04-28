# Project settings and configurations

# Paths
DATA_RAW_DIR = "Data_set_directory/Raw"
DATA_PROCESSED_DIR = "Data_set_directory/Processed"
MODELS_DIR = "Models_data"

# Model settings
DEFAULT_MODEL = "random_forest"
RANDOM_SEED = 42

# Feature extraction settings
MAX_FEATURES = 1000
NGRAM_RANGE = (1, 2)

# Training settings
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Suspicious keywords and patterns
PHISHING_KEYWORDS = [
    "account", "update", "verify", "login", "confirm", 
    "password", "security", "alert", "urgent", "suspend"
]

SUSPICIOUS_TLDS = [".info", ".xyz", ".top", ".tk", ".ml"]
