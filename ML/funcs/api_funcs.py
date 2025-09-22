import sys

# Function to handle unintended arguments
def get_target_arg():
    for arg in sys.argv:
        if arg.startswith('--target='):
            return arg.split('=')[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        return sys.argv[1]
    raise ValueError("No target feature provided. Please specify the target feature.")

def get_feature_addition_rounds_arg():
    for arg in sys.argv:
        if arg.startswith('--feature_addition_rounds='):
            return int(arg.split('=')[1])
    return 2  # Default to 5 rounds if not specified

def get_feature_dropping_threshold_arg():
    for arg in sys.argv:
        if arg.startswith('--feature_dropping_threshold='):
            return float(arg.split('=')[1])
    return 0.0002  # Default threshold

def get_tsfresh_fc_params_arg():
    for arg in sys.argv:
        if arg.startswith('--tsfresh_fc_params='):
            return arg.split('=')[1]
    return 'MinimalFCParameters'  # Default TSFRESH parameters