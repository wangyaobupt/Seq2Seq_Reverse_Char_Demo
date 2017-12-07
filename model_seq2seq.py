import tensorflow as tf
import numpy as np
import time
import helper
from tensorflow.python.layers.core import Dense

source_path = 'data/letters_source.txt'
target_path = 'data/letters_target.txt'
checkpoint_dir='./'
checkpoint = "best_model.ckpt"
tfWriterLogPath = './tf_writer'

# Number of Epochs
epochs = 240
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001
# attention units number
attention_num_units = 10

#Data preparation
source_sentences = helper.load_data(source_path)
target_sentences = helper.load_data(target_path)

#source_int_to_letter, source_letter_to_int = helper.extract_vocab(source_sentences)
source_int_to_letter, source_letter_to_int = helper.extract_alphabet_vocab()
#target_int_to_letter, target_letter_to_int = helper.extract_vocab(target_sentences)
target_int_to_letter, target_letter_to_int = helper.extract_alphabet_vocab()
  
source_letter_ids = [\
    [source_letter_to_int.get(letter, source_letter_to_int['<unk>']) for letter in line] \
    for line in source_sentences.split('\n')]
target_letter_ids = [
    [target_letter_to_int.get(letter, target_letter_to_int['<unk>']) for letter in line] + [target_letter_to_int['<\s>']] \
    for line in target_sentences.split('\n')] 

def get_model_inputs():
  input_data = tf.placeholder(tf.int32, [None, None], name='input')
  targets = tf.placeholder(tf.int32, [None, None], name='targets')
  lr = tf.placeholder(tf.float32, name='learning_rate')

  target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
  max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
  source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
  
  return input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length


def encoding_layer(input_data, rnn_size, num_layers,
                 source_sequence_length, source_vocab_size, 
                 encoding_embedding_size):
  # Encoder embedding
  enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)
  print 'source_vocab_size = {0}'.format(source_vocab_size)
  
  with tf.name_scope('encoder'):
      # RNN cell
      def make_cell(rnn_size):
          enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                             initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
          return enc_cell
    
      enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
      
      enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
  
  return enc_output, enc_state

# Process the input we'll feed to the decoder
def process_decoder_input(target_data, vocab_to_int, batch_size):
  '''Remove the last word id from each batch and concat the <s> to the begining of each batch'''
  ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
  dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<s>']), ending], 1)

  return dec_input
  
def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, enc_state, dec_input, attention_mechanism):
    # 1. Decoder Embedding
    target_vocab_size = len(target_letter_to_int)
    print 'target_vocab_size = {0}'.format(target_vocab_size)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    print 'dec_embed_input.shape={0}'.format(dec_embed_input.shape)

    # 2. Construct the decoder cell
    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    
    dec_cell_atten = tf.contrib.seq2seq.AttentionWrapper(
                  dec_cell, 
                  attention_mechanism,
                  attention_layer_size=attention_num_units)
     
    # 3. Dense layer to translate the decoder's output at each time 
    # step into a choice from the target vocabulary
    output_layer = Dense(target_vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))


    # 4. Set up a training decoder and an inference decoder
    # Training Decoder
    with tf.variable_scope("decode"):

        # Helper for the training process. Used by BasicDecoder to read inputs.
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        
        
        # Basic decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           enc_state,
                                                           output_layer) 
        
        # Perform dynamic decoding using the decoder
        training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)[0]
    # 5. Inference Decoder
    # Reuses the same parameters trained by the training process
    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<s>']], dtype=tf.int32), [batch_size], name='start_tokens')

        # Helper for the inference process.
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                start_tokens,
                                                                target_letter_to_int['<\s>'])

        # Basic decoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        enc_state,
                                                        output_layer)
        
        # Perform dynamic decoding using the decoder
        inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            impute_finished=True,
                                                            maximum_iterations=max_target_sequence_length)[0]
                                                            
        print 'inference_decoder_output.rnn_output = {0}'.format(inference_decoder_output[0].shape)
    
    return training_decoder_output, inference_decoder_output

def seq2seq_model(input_data, 
                  targets, 
                  learningRate, 
                  target_sequence_length, 
                  max_target_sequence_length, 
                  source_sequence_length,
                  source_vocab_size, 
                  target_vocab_size,
                  encoding_embedding_size, 
                  decoding_embedding_size, 
                  rnn_size, 
                  num_layers):
    
    enc_output, enc_state = encoding_layer(input_data, 
                                            rnn_size, 
                                            num_layers,
                                            source_sequence_length, 
                                            source_vocab_size, 
                                            encoding_embedding_size)
                                            
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                                            attention_num_units, 
                                            enc_output,
                                            memory_sequence_length=source_sequence_length)
    
    dec_input = process_decoder_input(targets, target_letter_to_int, batch_size)
    
    training_decoder_output, inference_decoder_output = decoding_layer(
                   target_letter_to_int, 
                   decoding_embedding_size, 
                   num_layers, 
                   rnn_size,
                   target_sequence_length, 
                   max_target_sequence_length, 
                   enc_state, 
                   dec_input,
                   attention_mechanism)
    
    return training_decoder_output, inference_decoder_output
    
    

# Build the graph as global variable
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():    
    # Load the model inputs    
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_model_inputs()
    
    # Create the training and inference logits
    training_decoder_output, inference_decoder_output = seq2seq_model(input_data, 
                                                                    targets, 
                                                                    lr, 
                                                                    target_sequence_length, 
                                                                    max_target_sequence_length, 
                                                                    source_sequence_length,
                                                                    len(source_letter_to_int),
                                                                    len(target_letter_to_int),
                                                                    encoding_embedding_size, 
                                                                    decoding_embedding_size, 
                                                                    rnn_size, 
                                                                    num_layers)    
    
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')
    
    # Create the weights for sequence_loss
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope("optimization"):
      
      # Loss function
      cost = tf.contrib.seq2seq.sequence_loss(
          training_logits,
          targets,
          masks)
    
      # Optimizer
      optimizer = tf.train.AdamOptimizer(lr)
    
      # Gradient Clipping
      gradients = optimizer.compute_gradients(cost)
      capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
      train_op = optimizer.apply_gradients(capped_gradients)
    
    # tensorboard  
    tb_sum = tf.summary.merge_all()


def train():
    # Split data to training and validation sets
    train_source = source_letter_ids[batch_size:]
    train_target = target_letter_ids[batch_size:]
    valid_source = source_letter_ids[:batch_size]
    valid_target = target_letter_ids[:batch_size]
    
    
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = \
      next(helper.get_batches(valid_target, valid_source, batch_size,
                           source_letter_to_int['<pad>'],
                           target_letter_to_int['<pad>']))

    display_step = 20 # Check training loss after every 20 batches

    with tf.Session(graph=train_graph) as sess:
        # Create Tensorboard writer
        helper.removeFileInDir(tfWriterLogPath)
        train_writer = tf.summary.FileWriter(tfWriterLogPath + '/train', sess.graph)
        train_writer.flush()
      
        all_vars = tf.global_variables()
        saver = tf.train.Saver(all_vars)
      
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model loaded from %s" % (ckpt.model_checkpoint_path))
        else:
            print("No checkpoint file found in {0}".format(checkpoint))
            sess.run(tf.global_variables_initializer())
        
        for epoch_i in range(1, epochs+1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                helper.get_batches(train_target, train_source, batch_size,
                           source_letter_to_int['<pad>'],
                           target_letter_to_int['<pad>'])):
            
                # Training step
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths})
                    
                # Debug message updating us on the status of the training
                if batch_i % display_step == 0 and batch_i > 0:
                    
                    # Calculate validation cost
                    validation_loss = sess.run(
                      [cost],
                      {input_data: valid_sources_batch,
                        targets: valid_targets_batch,
                        lr: learning_rate,
                        target_sequence_length: valid_targets_lengths,
                        source_sequence_length: valid_sources_lengths})
                    
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(train_source) // batch_size, 
                                  loss, 
                                  validation_loss[0]))

    
    
        # Save Model
        saver.save(sess, checkpoint_dir + checkpoint)
        print('Model Trained and Saved')

def source_to_seq(text):
    '''Prepare the text for the model'''
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<unk>']) for word in text]+ [source_letter_to_int['<pad>']]*(sequence_length-len(text))

def predict(input_sentence):
    text = source_to_seq(input_sentence)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint_dir + checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    
        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                      target_sequence_length: [len(text)]*batch_size, 
                                      source_sequence_length: [len(text)]*batch_size})[0] 


    pad = source_letter_to_int["<pad>"] 

    print('Original Text:', input_sentence)

    print('\nSource')
    print('  Word Ids:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))

    print('\nTarget')
    print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))

if __name__ == '__main__':
    #train()
    predict('jellyfish')
    #print source_sentences[:2]
    #print source_letter_ids[:2]
    #print source_letter_to_int