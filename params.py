import os

path = os.path.dirname(os.path.basename(__file__))


DATA_PATH = os.path.join(path, "inpainting")
TRAIN_PATH = "train2014"
VALID_PATH = "val2014"
CAPTION_PATH = "dict_key_imgID_value_caps_train_and_valid.pkl"

EXPERIMENT_PATH = os.path.join(path, "experiments/exp005")