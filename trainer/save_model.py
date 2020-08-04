#! /usr/bin/env python
import tensorflow as tf

meta_path = 'gs://ordinal-reason-282519-aiplatform/text_cnn_training_071320201841/checkpoints/model-22000.meta' # Your .meta file
output_node_names = ['output/predictions']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('gs://ordinal-reason-282519-aiplatform/text_cnn_training_071320201841/checkpoints/'))

    # Freeze the graph
#     output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print("output_node_names_1 : {}".format(output_node_names))
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('text_cnn3.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
