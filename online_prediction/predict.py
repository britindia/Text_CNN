#! /usr/bin/env python
# Commands to run this script
# ./predict.py --eval_train

import tensorflow as tf 
import numpy as np
import os
import time

import datetime
import data_helpers
# from trainer.data_helpers import batch_iter
# from model import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
# raw_input has the review entered by user
tf.flags.DEFINE_string("raw_input", "/tmp/input_reviews.csv", "Raw data for reviews entered Online.")
tf.flags.DEFINE_string("bucket", "gs://ordinal-reason-282519-aiplatform/text_cnn_training_071320201841", "Current directory.")
# model_dir=os.path.join(os.path.curdir,"..","trainer")
tf.flags.DEFINE_string("model_dir", os.path.join(os.path.curdir,"..","trainer"), "Current directory.")
os.path.curdir

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("raw_text", "", "text entered by the user")
# tf.flags.DEFINE_string("checkpoint_dir", os.path.join(FLAGS.bucket,"checkpoints/"), "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
import sys
FLAGS(sys.argv)

print("raw_text is: {}".format(FLAGS.raw_text))
print("raw_input: {}".format(FLAGS.raw_input))

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# eval_train will be TRUE if run through flask app. Use below command to run this script standalone  - 
# ./predict.py --raw_text="<your review>"
if FLAGS.eval_train:
    raw_reviews = list(open(FLAGS.raw_input, "r", encoding='utf-8').readlines())
    x_raw = [s.strip() for s in raw_reviews]
#     os.close(FLAGS.raw_input)
    os.remove(os.path.join("/tmp/","input_reviews.csv"))
else:
#     x_raw = ["a masterpiece four years in the making", "everything is off."]
    x_raw = [FLAGS.raw_text]
#     y_test = [0]

# Map data into vocabulary
# vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
# vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_path = os.path.join(FLAGS.bucket, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.bucket,"checkpoints/"))
# checkpoint_file = tf.train.latest_checkpoint(os.path.join(os.path.curdir,"/runs/1593655453/checkpoints/"))
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            if batch_predictions == 0:
               sentiment = ["Negative"]
            else:
               sentiment = ["Positive"]
            
            all_predictions = np.concatenate([all_predictions, sentiment])

# skipping accuracy test in this version. 
# Print accuracy if y_test is defined
# if y_test is not None:
#     correct_predictions = float(sum(all_predictions == y_test))
#     print("Total number of test examples: {}".format(len(y_test)))
#     print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
out_path=os.path.join("/tmp/", "online_prediction.csv")
# if os.path.exists(out_path):
# #    open(os.path.join("/tmp/","input_reviews.csv"))
#    os.remove(out_path)
#    print("Prediction File Removed")
# else:
#    open(out_path, 'w').close()
#    print("Prediction File Created")
print("Saving evaluation to {0}".format(out_path))
print("Saving predictions_human_readable to {0}".format(predictions_human_readable))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
