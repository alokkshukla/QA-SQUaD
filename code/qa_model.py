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
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
#import tf.nn.rnn_cell

#from tensorflow.nn import sparse_softmax_cross_entropy_with_logits as ssce 

ssce = tf.nn.sparse_softmax_cross_entropy_with_logits

from evaluate import exact_match_score, f1_score
from util import Progbar, minibatches

from tensorflow.python.ops.nn import bidirectional_dynamic_rnn 

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class FilterLayer(object): 
    def __init__(self, ):
        self.size = size
        self.vocab_dim = vocab_dim

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
    def __init__(self, q_encoder, p_encoder, decoder, embed_path, question_length, paragraph_length, flags, *args):
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
        self.config = flags

        # ==== set up placeholder tokens ========
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, question_length))
        self.paragraph_placeholder = tf.placeholder(tf.int32, shape=(None, paragraph_length))
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None, 2))

        self.qlen_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.plen_placeholder = tf.placeholder(tf.int32, shape=(None))


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

        # WORD EMBEDDINGS
        # (minibatch size, question length, embedding size)
        q = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)
        p = tf.nn.embedding_lookup(self.embeddings, self.paragraph_placeholder)

        # FILTER LAYER w00t
        def l2_norm(tensor, indices=None):
            return tf.sqrt(tf.reduce_sum(tf.square(tensor), reduction_indices=indices, keep_dims=True))

        q_mags = l2_norm(q, [2])
        p_mags = l2_norm(p, [2])
        q_normalized = tf.truediv(q,q_mags)
        p_normalized = tf.truediv(p,p_mags)

        r = tf.einsum('aik,ajk->aij', q_normalized, p_normalized)
        rmax = tf.reduce_max(r, reduction_indices=[1])

        #p = tf.mul(p, rmax)
        p = tf.einsum('aij,ai->aij', p, rmax)

        # CONTEXT REPRESENTATION LAYER uhhh

        with tf.variable_scope("context"):
            # Note. We're using the same cell. Maybe not?
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.state_size)

            # (minibatch size, max question length, hidden state size)
            with tf.variable_scope("bilstm1"):
                (fwQ, bwQ), _ = bidirectional_dynamic_rnn(cell, cell, q,
                                self.qlen_placeholder, dtype=np.float32)

            with tf.variable_scope("bilstm2"):
                (fwP, bwP), _ = bidirectional_dynamic_rnn(cell, cell, p,
                                self.plen_placeholder, dtype=np.float32)

        # MULTI PERSPECTIVE CONTEXT MATCHING LAYER
        xavier = tf.contrib.layers.xavier_initializer()

        # Let's do naive cosine instead
        fwQnorm = tf.truediv(fwQ, l2_norm(fwQ, [2]))
        bwQnorm = tf.truediv(bwQ, l2_norm(bwQ, [2]))

        fwPnorm = tf.truediv(fwP, l2_norm(fwP, [2]))
        bwPnorm = tf.truediv(bwP, l2_norm(bwP, [2]))

        # (batch size, question length, paragraph length)
        fwSim = tf.einsum('aij,akj->aik', fwQnorm, fwPnorm)
        bwSim = tf.einsum('aij,akj->aik', bwQnorm, bwPnorm)


        # Okay so here's a fun bug!
        # fwSim and bwSim include a bunch of NaNs because each seq. has
        # a different elngth. We need to perform those three operations
        # (first/last, max, mean) only up to [:qlen], [:plen], in a way.
        # This seems to be the idea behind pooling but tf is being weird

        # Matching full similarity

        def get_last(a):
            mat = a[0]
            ln = a[1]
            return mat[ln-1, :]

        def get_first(a):
            mat = a[0]
            ln = a[1]
            return mat[0, :]

        def get_max(a):
            ln = a[1]
            mat = a[0][:ln, :]
            return tf.reduce_max(mat, 0)

        def get_mean(a):
            ln = a[1]
            mat = a[0][:ln, :]
            return tf.reduce_mean(mat, 0)

        FULLfw = tf.map_fn(get_last, (fwSim, self.qlen_placeholder), dtype=np.float32)
        FULLbw = tf.map_fn(get_first, (bwSim, self.qlen_placeholder), dtype=np.float32)

        self.thing = FULLfw
        #fwSim[0, self.qlen_placeholder[0]-1, self.plen_placeholder[0]-1]
        #fwSim[:, tf.expand_dims(self.qlen_placeholder,1)-1, :]


        MAXfw = tf.map_fn(get_max, (fwSim, self.qlen_placeholder), dtype=np.float32)
        MAXbw = tf.map_fn(get_max, (bwSim, self.qlen_placeholder), dtype=np.float32)

        MEANfw = tf.map_fn(get_mean, (fwSim, self.qlen_placeholder), dtype=np.float32)
        MEANbw = tf.map_fn(get_mean, (bwSim, self.qlen_placeholder), dtype=np.float32)

        # (batch size, passage length, 6)
        ms = tf.pack([FULLfw, FULLbw, MAXfw, MAXbw, MEANfw, MEANbw], axis=2)
        print("ms shape", ms.get_shape())
        # AGGREGATION LAYER
        with tf.variable_scope("aggregation"):
            # fwA, bwA: (batch size, passage length, aggregation hidden size)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.aggregation_size)
            (fwA, bwA), _ = bidirectional_dynamic_rnn(cell, cell, ms,
                            self.plen_placeholder, dtype=np.float32)

            # (batch size, passage length, 2*aggregation hidden size)
            mergedA = tf.concat(2, [fwA, bwA])

        # PREDICTION LAYER
        

        return
        """
        # 1. Full matching perspective
        # - SLOW AF -
        def get_m(v1, v2, name):
            ms = []
            with tf.variable_scope(name):
                try:
                    W = tf.get_variable("W", shape=(self.config.perspectives,
                                                self.config.state_size),
                                    initializer=xavier)
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    W = tf.get_variable("W")

                tr = tf.einsum('ij,aj->iaj', W, v1)
                for k in range(self.config.perspectives):
                    v1k = tf.einsum('ai,i->ai', v1, W[k,:])
                    v1k = tf.truediv(v1k, l2_norm(v1k, [1]))

                    v2k = tf.einsum('ai,i->ai', v2, W[k,:])
                    v2k = tf.truediv(v2k, l2_norm(v2k, [1]))

                    ms.append(tf.reduce_sum(tf.mul(v1k, v2k),axis=1))
            return tf.pack(ms, 1)
        fullMatchingFw = []
        fullMatchingBw = []
        for j in range(self.paragraph_length):
            print(j)
            fullMatchingFw.append(get_m(fwP[:,j,:], fwQ[:,self.question_length-1,:],"fmfw"))
            fullMatchingBw.append(get_m(bwP[:,j,:], fwQ[:,0,:],"fmbw"))
        fullMatchingFw = tf.pack(fullMatchingFw, 1)
        fullMatchingBw = tf.pack(fullMatchingBw, 1)
        print("hmmm", fullMatchingFw.get_shape())"""

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            #l1 = ssce(self.a_s, self.start_answer)
            #l2 = ssce(self.a_e, self.end_answer)
            self.loss = tf.constant(42)#l1 + l2 


    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            pretrained_embeddings = np.load(self.embed_path)['glove'].astype(np.float32)
            # TODO variable
            self.embeddings = tf.constant(pretrained_embeddings)

    def make_feed_dict(self, paragraphs, questions, labels):
        # Given a batch
        dic = {}
        def fill(L):
            def _fill(x):
                x = x[:L]
                for i in range(L-len(x)):
                    x.append(0)
                return x
            return _fill

        dic[self.question_placeholder] = np.array(map(fill(self.question_length), questions))
        dic[self.paragraph_placeholder] = np.array(map(fill(self.paragraph_length), paragraphs))

        dic[self.plen_placeholder] = np.array(map(len, paragraphs))
        dic[self.qlen_placeholder] = np.array(map(len, questions))

        dic[self.label_placeholder] = labels

        return dic

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

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

        answer = p[a_s, a_e + 1]
        true_answer = p[true_s, true_e + 1]
        f1 = f1_score(answer, true_answer)
        em = exact_match_score(answer, true_answer)

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train_on_batch(self, session, batch):
        #print("Training on batch", batch)
        feed = self.make_feed_dict(*batch)
        output = session.run([self.thing], feed_dict=feed)
        print("output is",output)
        exit()
        return loss

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
