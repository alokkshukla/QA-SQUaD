from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from os.path import join as pjoin

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from util import Progbar, minibatches

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, max_length, size, vocab_dim, batch_size):
        self.max_length = max_length
        self.size = size
        self.vocab_dim = vocab_dim
        self.batch_size = batch_size

    def encode(self, inputs, lengths, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        outputs, last_states = tf.nn.dynamic_rnn(lstm, sequence_length=lengths, inputs=inputs, dtype=np.float32)
        print("SHAPE", outputs.get_shape())
        """state = tf.zeros([self.batch_size, self.size])
        print(tf.shape(state))
	for word in range(self.max_length):
	    output, state = lstm(inputs[:, word], state)
            print(tf.shape(output), tf.shape(state))"""

        return outputs


class Decoder(object):
    def __init__(self, output_size, hidden_size, q_len):
        self.output_size = output_size
        self.q_len = q_len

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        Hq, Hp = knowledge_rep
        m = tf.matmul
        # hip = h_i^p
        # hap = h_{i-1}^r
        eQ = tf.ones(self.q_len)
        l = self.hidden_size
        init = tf.contrib.layers.xavier_initializer()
        Wq = tf.get_variable("Wq", (l, l), initializer=init)
        Wp = tf.get_variable("Wp", (l, l), initializer=init)
        Wr = tf.get_variable("Wr", (l, l), initializer=init)
        b = tf.get_variable("b", ())
        bp = tf.get_variable("bp", (l), initializer=init)
        w = tf.get_variable("w", (1,l), initializer=init)
        for i in range(self.q_len):
            Gi = tf.tanh(m(Wq, Hq)+tf.outer(m(Wp, hip)+m(Wr,hap)+bp, eQ))
            alphai = tf.softmax(m(w, Gi)+tf.outer(b, eQ))

        return

class QASystem(object):
    def __init__(self, q_encoder, p_encoder, decoder, embed_path, question_length, paragraph_length, batch_size, flags, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        self.embed_path = embed_path
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder
        self.question_length = question_length
        self.paragraph_length = paragraph_length
        self.batch_size = batch_size
        self.config = flags

        # ==== set up placeholder tokens ========
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, question_length))
        self.qlen_placeholder = tf.placeholder(tf.bool, shape=(None))
        self.qmask_placeholder = tf.placeholder(tf.bool, shape=(None, question_length))
        self.paragraph_placeholder = tf.placeholder(tf.int32, shape=(None, paragraph_length))
        self.plen_placeholder = tf.placeholder(tf.bool, shape=(None))
        self.pmask_placeholder = tf.placeholder(tf.bool, shape=(None, paragraph_length))
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None, 2))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        q = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)
        par = tf.nn.embedding_lookup(self.embeddings, self.paragraph_placeholder)

        with vs.variable_scope("q"):
            q_enc = self.q_encoder.encode(q, self.qlen_placeholder, None)
        with vs.variable_scope("p"):
            p_enc = self.p_encoder.encode(par, self.plen_placeholder, None)

        self.answer = q_enc
        #dec = self.decoder.decode((q_enc, p_enc))


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            pretrained_embeddings = np.load(self.embed_path)['glove'].astype(np.float32)
            # TODO varible
            self.embeddings = tf.constant(pretrained_embeddings)

    def make_feed_dict(self, datapoints):
        # Given a batch
        dic = {}
        def fill(L):
            def _fill(x):
                x = x[:L]
                for i in range(L-len(x)):
                    x.append(0)
                return x
            return _fill

        questions = [datapoint[1] for datapoint in datapoints]
        dic[self.qlen_placeholder] = map(len, questions)
        dic[self.question_placeholder] = map(fill(self.question_length), questions)
        paragraphs = [datapoint[0] for datapoint in datapoints]
        dic[self.plen_placeholder] = map(len, paragraphs)
        dic[self.paragraph_placeholder] = map(fill(self.paragraph_length), paragraphs)
        return dic


    def optimize(self, session, batch):#train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        #input_feed = {}

        # fill in this feed_dictionary like:
        #input_feed['train_x'] = train_x
        input_feed = self.make_feed_dict(batch)

        output_feed = [self.answer]

        outputs = session.run(output_feed, input_feed)
        print("outputs!")
        print(outputs)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train_on_batch(self, session, batch):
        #print("Training on batch", batch)
        #_, loss = session.run([self.loss], feed_dict=feed)
        #return loss
        return 42

    def run_epoch(self, session, dataset, val_dataset):
        prog = Progbar(target=1 + int(len(dataset) / self.config.batch_size))
        for i, batch in enumerate(minibatches(dataset, self.config.batch_size)):
            loss = self.train_on_batch(session, batch)
            prog.update(i+1, [("train loss", loss)])
        print("")

        f1, em = self.evaluate_answer(session, val_dataset)
        return f1 # return validation f1

    def train(self, session, dataset, val_dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """
        """for epoch in range(10):
            print("Epoch", epoch)
            for i in range(20):
                ind = list(np.random.choice(len(dataset), self.batch_size, replace=False))
                self.optimize(session, map(lambda x: dataset[x], ind))"""

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        logging.info("Starting training. Nominally %d epochs." % self.config.epochs)
        best_score = -1.0
        for epoch in range(self.config.epochs):
            logging.info("Doing epoch %d", epoch + 1)
            score = self.run_epoch(session, dataset, val_dataset)
            if score > best_score:
                fn = pjoin(train_dir,"model.weights")
                logging.info("New best score! Saving model in %s.something" % fn)
                self.saver.save(session, fn)
                best_score = score
            print("")

