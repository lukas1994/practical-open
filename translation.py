import random
import itertools
import numpy as np
import tensorflow as tf

import seq2seq_model

TRAIN_PATH = 'iwslt-deen-prepared/train.de-en'
TEST_PATH = 'iwslt-deen-prepared/test.de-en'
VALID_PATH = 'iwslt-deen-prepared/valid.de-en'

TRAIN_DIR = 'train'

# Special vocabulary symbols - we always put them at the start.
PAD = b"_PAD"
GO = b"_GO"
EOS = b"_EOS"
UNK = b"_UNK"
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def tokenize(line):
    [german, english] = line.split('|||')
    german = german.split(' ')
    english = english.split(' ')
    return german, english

def read_file(file):
    with open(file) as f:
        data = f.read().splitlines()
    return list(map(tokenize, data))

class Vocab(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.id_map = {}
        for i, word in enumerate(vocab):
            self.id_map[word] = i

    def get_id(self, word):
        return self.id_map.get(word, UNK_ID)

    def size(self):
        return len(self.vocab)

def get_vocab(data):
    return Vocab(START_VOCAB + list(set(list(itertools.chain.from_iterable(data)))))

def add_unknown(data):
    for tokens in data:
        for i in range(len(tokens)):
            if random.random() < 0.001:
                tokens[i] = UNK
    return data

def map_to_ids(data, vocab_german, vocab_english):
    res = []
    for sample in data:
        res.append((
            list(map(vocab_german.get_id, sample[0])),
            list(map(vocab_english.get_id, sample[0]))
        ))
    return res

def get_data():
    train_data = read_file(TRAIN_PATH)
    [german, english] = [list(t) for t in zip(*train_data)]
    vobab_german = get_vocab(german)
    vobab_english = get_vocab(english)
    german = add_unknown(german)
    english = add_unknown(english)

    train_data = map_to_ids(list(zip(german, english)), vobab_german, vobab_english)
    test_data = map_to_ids(read_file(TEST_PATH), vobab_german, vobab_english)
    valid_data = map_to_ids(read_file(VALID_PATH), vobab_german, vobab_english)

    # print(np.histogram(list(map(lambda a: len(a), german))))

    return train_data, test_data, valid_data, vobab_german, vobab_english

def get_multiple_batches(data, n, batch_size):
    sample = random.sample(data, n*batch_size)
    sample.sort(key=lambda x: len(x[0]))
    batches = []

    for i in range(0, n*batch_size, batch_size):
        batch = sample[i:i+batch_size]
        encoder_size = max([len(x[0]) for x in batch])
        decoder_size = max([len(x[1]) for x in batch])
        encoder_inputs, decoder_inputs = [], []

        for x in batch:
            encoder_input = x[0]
            decoder_input = x[1]

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

        batches.append((batch_encoder_inputs, batch_decoder_inputs, batch_weights))
    return batches


def train(train_data, valid_data, vobab_german, vobab_english):
    BATCH_SIZE = 64
    LAYER_SIZE = 1024
    NUM_LAYERS = 1
    MAX_GRADIENT_NORM = 5.0
    LEARNING_RATE = 0.5
    LEARNING_RATE_DECAY_FACTOR = 0.99

    model = seq2seq_model.Seq2SeqModel(
        vobab_german.size(),
        vobab_english.size(),
        [(40, 40)], # max input sizes
        LAYER_SIZE,
        NUM_LAYERS,
        MAX_GRADIENT_NORM,
        BATCH_SIZE,
        LEARNING_RATE,
        LEARNING_RATE_DECAY_FACTOR,
        use_lstm=True,
    )

    print("model initialized...")

    with tf.Session() as session:
        ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Get a batch and make a step.
            start_time = time.time()

            for encoder_inputs, decoder_inputs, target_weights in get_multiple_batches(train_data, 20, BATCH_SIZE):

                _, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                           target_weights, 0, False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
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
                    checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                    model.saver.save(session, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    # Run evals on development set and print their perplexity.
                    encoder_inputs, decoder_inputs, target_weights = get_multiple_batches(valid_data, 1, BATCH_SIZE)[0]
                    _, eval_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                               target_weights, 0, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                      "inf")
                    print("  eval: perplexity %.2f" % eval_ppx)
                    sys.stdout.flush()

if __name__ == '__main__':
    train_data, test_data, valid_data, vobab_german, vobab_english = get_data()
    
    train(train_data, valid_data, vobab_german, vobab_english)
