import sys
sys.path.append("./")
from igcn.model import GCN
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

from igcn.utils import *
tf.flags.DEFINE_string('f','','')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("train_list", "../Data/train_list.txt", "train list.")
flags.DEFINE_string("test_list", "../Data/test_list.txt", "test list.")
flags.DEFINE_float("learning_rate", 1e-4, "Initial learning rate.")
flags.DEFINE_float("dropout", 0.5, "Dropout rate (1 - keep probability).") #not used
flags.DEFINE_integer("epochs", 300, "Number of epochs to train.")
flags.DEFINE_integer("hidden",256, "Number of units in hidden layer.") 
flags.DEFINE_integer("feat_dim", 6, "Number of units in feature layer.")
flags.DEFINE_integer("coord_dim", 3, "Number of units in output layer.")
flags.DEFINE_float("weight_decay", 5e-6, "Weight decay for L2 loss.")

# Specify target organ
organ = "liver"
nNode = 500
            
# Set random seed
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

num_supports = 1

# Define placeholders(dict) and model
placeholders = {
    "features": tf.placeholder(tf.float32, shape=(None, 3)),        # initial 3D coordinates
    "labels": tf.placeholder(dtype=tf.float32, shape=(None, 3)),          # ground truth
    "img_inp": tf.placeholder(tf.float32, shape=(640, 640, 3)),     # initial projection + DRR
    "img_label": tf.placeholder(tf.float32, shape=(640, 640, 3)),   # target deformation map
    "shapes": tf.placeholder(tf.float32, shape=(None, 3)),          # relative positions
    "ipos": tf.placeholder(tf.float32, shape=(None, 2)),            # initial projected points
    "adj" : tf.placeholder(tf.float32, shape=(nNode,nNode)),        # adj matrix size
    "rmax": tf.placeholder(tf.float32),                             # rmax for projection        
    "face": tf.placeholder(tf.int32, shape=(None, 4)),              # triangle face
    "face_norm": tf.placeholder(tf.float32, shape=(None, 3)),       # face normal vector
    "support": [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # for convolution
    "num_features_nonzero": tf.placeholder(tf.int32),                            # helper variable for sparse dropout
}

# Create model
model = GCN(placeholders, logging=True)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
# configuration options for a session
config = tf.ConfigProto()
# allows soft device placement and
config.allow_soft_placement=True
config.gpu_options.allow_growth = True
# Define tf session, an object called launch operations
sess = tf.Session(config=config)
### This works as expected
# x = tf.random.uniform(dtype=tf.float32, shape=(1, 3))
# print(sess.run(x))
adj = np.random.uniform(size=(nNode,nNode)).astype(np.float32)
variables_dict = {
    "features": np.random.uniform(size=(1, 3)).astype(np.float32),        # initial 3D coordinates
    "labels": np.random.uniform(size=(1, 3)).astype(np.float32),          # ground truth
    "img_inp": np.random.uniform(size=(640, 640, 3)).astype(np.float32),     # initial projection + DRR
    "img_label": np.random.uniform(size=(640, 640, 3)).astype(np.float32),   # target deformation map
    "shapes": np.random.uniform(size=(1, 3)).astype(np.float32),          # relative positions
    "ipos": np.random.uniform(size=(1, 2)).astype(np.float32),            # initial projected points
    "adj" : normalize_adj(adj),        # adj matrix size
    "rmax": np.random.uniform(size=(1,)).astype(np.float32),                             # rmax for projection        
    "face": np.random.uniform(size=(1, 4)).astype(np.int32),              # triangle face
    "face_norm": np.random.uniform(size=(1, 3)).astype(np.float32),       # face normal vector
    "support": [preprocess_adj(adj)], # for convolution
}

feed_dict = construct_feed_dict(**variables_dict, placeholders=placeholders)

# Assigns values to tf variables objects
sess.run(tf.global_variables_initializer())
_out = sess.run([model.outputs], feed_dict=feed_dict)
sess.close()
