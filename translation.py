import sys
import random
import time
import math
import itertools
import numpy as np
import tensorflow as tf

import seq2seq_model

TRAIN_PATH = "iwslt-deen-prepared/train.de-en"
TEST_PATH = "iwslt-deen-prepared/test.de-en"
VALID_PATH = "iwslt-deen-prepared/valid.de-en"

TRAIN_DIR = 'train'

# Special vocabulary symbols - we always put them at the start.
PAD = "_PAD"
GO = "_GO"
EOS = "_EOS"
UNK = "_UNK"
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# params
BATCH_SIZE = 64
LAYER_SIZE = 128 #1024
NUM_LAYERS = 1
MAX_GRADIENT_NORM = 5.0
LEARNING_RATE = 0.5
LEARNING_RATE_DECAY_FACTOR = 0.99
MAX_SENTENCE_LENGTH = 40
MAX_VOCAB_SIZE = 5000
STEPS_PER_CHECKPOINT = 1000

def tokenize(line):
    [german, english] = line.split('|||')
    german = german.strip().split(' ')
    english = english.strip().split(' ')
    return german, english

def read_file(file):
    with open(file) as f:
        data = f.read().splitlines()
    return list(map(tokenize, data))

class Vocab(object):
    def __init__(self, data=None):
        if not data: return

        freq = {}
        for word in data:
            freq[word] = freq.get(word, 0) + 1
        self.vocab = START_VOCAB + sorted(freq, key=freq.get, reverse=True)

        self.vocab = self.vocab[:MAX_VOCAB_SIZE]
        self.__generate_id_map()

    @staticmethod
    def __path(name):
        return 'train/' + name + '.txt'

    def __generate_id_map(self):
        self.id_map = {}
        for i, word in enumerate(self.vocab):
            self.id_map[word] = i

    @staticmethod
    def load(name):
        v = Vocab()
        with open(Vocab.__path(name)) as f:
            v.vocab = f.read().splitlines()
        v.__generate_id_map()
        return v

    def get_id(self, word):
        return self.id_map.get(word, UNK_ID)

    def get_word(self, i):
        return self.vocab[i]

    def size(self):
        return len(self.vocab)

    def write(self, name):
        with open(Vocab.__path(name), 'w') as f:
            f.write('\n'.join(self.vocab))

def add_unknown(data):
    for tokens in data:
        for i in range(len(tokens)):
            if random.random() < 0.001:
                tokens[i] = UNK
    return data

def map_to_ids(sentence, vocab):
    return list(map(vocab.get_id, sentence))

def map_data_to_ids(data, vocab_german, vocab_english):
    res = []
    for sample in data:
        res.append((
            map_to_ids(sample[0], vocab_german),
            map_to_ids(sample[1], vocab_english)
        ))
    return res

def get_data():
    train_data = read_file(TRAIN_PATH)

    [german, english] = [list(t) for t in zip(*train_data)]
    vocab_german = Vocab(itertools.chain.from_iterable(german))
    vocab_english = Vocab(itertools.chain.from_iterable(english))

    vocab_german.write('german')
    vocab_english.write('english')

    train_data = map_data_to_ids(train_data, vocab_german, vocab_english)
    test_data = map_data_to_ids(read_file(TEST_PATH), vocab_german, vocab_english)
    valid_data = map_data_to_ids(read_file(VALID_PATH), vocab_german, vocab_english)

    return train_data, test_data, valid_data

def get_batch(batch):
    batch_size = len(batch)
    encoder_size = MAX_SENTENCE_LENGTH
    decoder_size = MAX_SENTENCE_LENGTH
    encoder_inputs, decoder_inputs = [], []

    for encoder_input, decoder_input in batch:
        if len(encoder_input) > MAX_SENTENCE_LENGTH: encoder_input = encoder_input[:MAX_SENTENCE_LENGTH]
        if len(decoder_input) > MAX_SENTENCE_LENGTH-1: decoder_input = decoder_input[:MAX_SENTENCE_LENGTH-1]

        # Encoder inputs are padded and then reversed.
        encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([GO_ID] + decoder_input +
                              [PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in range(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in range(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

def get_multiple_batches(data, n, batch_size):
    sample = random.sample(data, n*batch_size)
    sample.sort(key=lambda x: len(x[0]))
    batches = []

    for i in range(0, n*batch_size, batch_size):
        batch = sample[i:i+batch_size]
        batches.append(get_batch(batch))

    return batches

def create_model(session, forward_only):
    model = seq2seq_model.Seq2SeqModel(
        MAX_VOCAB_SIZE, # german
        MAX_VOCAB_SIZE, # english
        [(MAX_SENTENCE_LENGTH, MAX_SENTENCE_LENGTH)], # max input sizes
        LAYER_SIZE,
        NUM_LAYERS,
        MAX_GRADIENT_NORM,
        BATCH_SIZE,
        LEARNING_RATE,
        LEARNING_RATE_DECAY_FACTOR,
        # use_lstm=True,
        forward_only=forward_only,
    )

    ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model

def train(train_data, valid_data):
    with tf.Session() as session:
        model = create_model(session, False)

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Get a batch and make a step.
            start_time = time.time()

            for encoder_inputs, decoder_inputs, target_weights in get_multiple_batches(train_data, 20, BATCH_SIZE):

                _, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                           target_weights, 0, False)
                step_time += (time.time() - start_time) / STEPS_PER_CHECKPOINT
                loss += step_loss / STEPS_PER_CHECKPOINT
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % STEPS_PER_CHECKPOINT == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print ("global step %d learning rate %.4f step-time %.2f perplexity "
                           "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                     step_time, perplexity))
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                      session.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)
                    # Save checkpoint and zero timer and loss.
                    checkpoint_path = "train/translate.ckpt"
                    model.saver.save(session, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    # Run evals on validation set and print their perplexity (one batch).
                    encoder_inputs, decoder_inputs, target_weights = get_multiple_batches(valid_data, 1, BATCH_SIZE)[0]
                    _, eval_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                               target_weights, 0, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("  eval: perplexity %.2f" % eval_ppx)
                    sys.stdout.flush()

def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        vocab_german = Vocab.load('german')
        vocab_english = Vocab.load('english')

        # Decode from standard input.
        def ask():
            sys.stdout.write("> ")
            sys.stdout.flush()
            return sys.stdin.readline().splitlines()[0]

        sentence = ask()
        while sentence != "":
            # Get token-ids for the input sentence.
            token_ids = map_to_ids(sentence.split(" "), vocab_german)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = get_batch([(token_ids, [])])

            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, 0, True)

            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

            # If there is an EOS symbol in outputs, cut them at that point.
            if EOS_ID in outputs:
                print("EOS found")
                outputs = outputs[:outputs.index(EOS_ID)]

            # Print out result sentence.
            print(" ".join([vocab_english.get_word(output) for output in outputs]))

            sentence = ask()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        train_data, test_data, valid_data = get_data()

        train(train_data, valid_data)
    else:
        decode()
