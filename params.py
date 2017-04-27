import os

path = os.path.dirname(os.path.basename(__file__))

# data set path
DATA_PATH = os.path.join(path, "inpainting")
TRAIN_PATH = "train2014"
VALID_PATH = "val2014"
CAPTION_PATH = "dict_key_imgID_value_caps_train_and_valid.pkl"

BATCH_SIZE = 32
NB_EPOCHS = 150
# patience for early stopping
PATIENCE = 15
# path to save logs and model
EXPERIMENT_PATH = os.path.join(path, "experiments/gan022")
# whether to use dropout
USE_DROPOUT = False

# Parameters of adversarial loss
USE_ADVERSARIAL_LOSS = True
LAMBDA_DECAY = False
LAMBDA_ADVERSARIAL = .998
DISCR_WHOLE_IMAGE = True
DISCR_LOSS_LIMIT = .2
