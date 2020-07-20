import numpy as np
import re
import os
from six.moves import urllib
import tempfile
import tensorflow as tf


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def _download_and_clean_file(filename, input_dir):
    """Downloads data from Google Storage or simply copy to /tmp/ directory in case of
       local run 

    Args:
      filename: filename to save input_dir to
      url: bucket url in case of file hosted on GCP
    """
    
    dataset=tf.data.TextLineDataset(input_dir)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    next_element = iterator.get_next()
    with tf.io.gfile.GFile(filename, 'w') as file_object:
       
       with tf.Session() as sess:
            try:
               sess.run(init_op)
               while True:
                   line = sess.run(next_element)
                   file_object.write(line)
                   file_object.write('\r\n')
            except tf.errors.OutOfRangeError:
                   print("End of dataset")
                   print("{} saved at {}".format(input_dir,filename))
       
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    DATA_DIR = os.path.join(tempfile.gettempdir(),'Text_CNN')
    tf.io.gfile.makedirs(DATA_DIR)
    os.system('ls DATA_DIR')
    positive_data_file_target='rt-polarity.pos'
    negative_data_file_target='rt-polarity.neg'

    positive_data_file_output = os.path.join(DATA_DIR, positive_data_file_target)
    negative_data_file_output = os.path.join(DATA_DIR, negative_data_file_target)
    if tf.io.gfile.exists(positive_data_file_output):
       os.remove(positive_data_file_output)
    _download_and_clean_file(positive_data_file_output, positive_data_file)
    if tf.io.gfile.exists(negative_data_file_output):
       os.remove(negative_data_file_output)
    _download_and_clean_file(negative_data_file_output, negative_data_file)

    # Load data from files
    if tf.io.gfile.exists(positive_data_file_output):
       print("Positive File length : {}".format(len(open(positive_data_file_output).readlines(  ))))
    if tf.io.gfile.exists(negative_data_file_output):
       print("Negative File length : {}".format(len(open(negative_data_file_output).readlines(  ))))
    positive_examples = list(open(positive_data_file_output, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]

    negative_examples = list(open(negative_data_file_output, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y]
    

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    print ("Length of x_train + y_train data : {}".format(len(data)))
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    print ("num_batches_per_epoch : {}".format(num_batches_per_epoch))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
