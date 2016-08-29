#!/usr/bin/python
from chainer import Chain, Variable
import chainer.functions as F
import chainer.optimizers as O
from chainer.optimizer import WeightDecay
import chainer.links as L
import numpy as np
import random
import sys
# from tqdm import tqdm
import cPickle as pickle

# USE_GPU = True
USE_GPU = False

if USE_GPU:
    from chainer import cuda
    xp = cuda.cupy
else:
    import numpy as xp

#########################################################
################ Vocabulary, Token, Tree ################
#########################################################

class Vocab(object):
    unk = "UNKNOWN"
    pad = "PADDING"
    unk_id = 0
    pad_id = 1

    def __init__(self, word_path, tag_path, label_path, pretrained_path=None, cutoff=1):
        self.words  = Vocab._read_listfile(word_path)
        self.tags   = Vocab._read_listfile(tag_path, init={Vocab.unk: 0})
        self.labels = Vocab._read_listfile(label_path, init={Vocab.unk: 0})
        self.freq   = None
        self.cutoff = cutoff
        self._accept_new_entries = True

        if pretrained_path is not None:
            self.pretrained = Vocab._read_pretrained(pretrained_path)
        else:
            self.pretrained = None

    @staticmethod
    def _freqdict(path):
        res = {Vocab.unk: 0}
        for line in open(path):
            line = line.strip()
            if len(line) == 0: continue
            items = line.split("\t")
            word = Vocab.normalize(items[1])
            if res.has_key(word):
                res[word] += 1
            else:
                res[word] = 0
        return res

    @staticmethod
    def _read_listfile(path, init={}):
        res = init
        for line in open(path):
            res[line.strip()] = len(res)
        return res

    @staticmethod
    def normalize(word):
        n = word.lower()
        if n == "-lrb-":
            return "("
        elif n == "-rrb-":
            return ")"
        else:
            return n

    def _get_or_add_entry(self, entry):
        if self.words.has_key(entry):
            return self.words[entry]
        elif self._accept_new_entries:
            idx = len(self.words)
            self.words[entry] = idx
            return idx
        else:
            return Vocab.unk_id

    def targetsize(self):
        # NOOP: 0
        # SHIFT: 1
        # REDUCEL(i) >=2 even, i in [1 to labelsize]
        # REDUCER(i) >=3 odd, i in [1 to labelsize]
        # REDUCE(0) == REDUCE(UNKNOWN) is ignored
        nlabels = len(self.labels) - 1
        return 2 + 2 * nlabels

    def read_conll_train(self, filepath):
        self._accept_new_entries = True
        if self.freq is None:
            self.freq = self._freqdict(filepath)
        res = self._read_conll(filepath)
        if self.pretrained is not None:
            vsize, dim = self.pretrained.shape
            new_pretrained = np.ndarray((len(self.words), dim), 'f')
            new_pretrained[:vsize, :] = self.pretrained

            for i in range(vsize, len(self.words)):
                # initialize with [0.01,-0.01] normal distribution
                new_pretrained[i, :] = 0.02 * np.random.random_sample() - 0.01
            self.pretrained = new_pretrained
        return res

    def read_conll_test(self, filepath):
        self._accept_new_entries = False
        res = self._read_conll(filepath)
        return res

    def _read_conll(self, filepath):
        res = []
        words, tags   = [Vocab.pad], [Vocab.unk]
        labels, heads = [Vocab.unk], [0]
        for line in open(filepath, "r"):
            line = line.strip()
            if line == "":
                w = [self._get_or_add_entry(
                    Vocab.normalize(w)) for w in words]
                t = [self.tags[t] for t in tags]
                l = [self.labels[l] for l in labels]
                res.append(Sentence(
                    words, tags, labels, heads, w, t, l))
                words, tags   = [Vocab.pad], [Vocab.unk]
                labels, heads = [Vocab.unk], [0]
                continue
            items = line.split("\t")
            words.append(items[1])
            tags.append(items[4])
            labels.append(items[7])
            heads.append(int(items[6]))
        return filter(lambda s: len(s) > 1, res)

    @staticmethod
    def _read_pretrained(filepath):
        io = open(filepath)
        dim = len(io.readline().split())
        io.seek(0)
        nvocab = sum(1 for line in io)
        io.seek(0)
        res = np.empty((nvocab, dim), dtype=np.float32)
        for i, line in enumerate(io):
            line = line.strip()
            if len(line) == 0: continue
            res[i] = line.split()
        io.close()
        return res

    def _read_probs(self, filepath, top=3):
        root = [.0] * len(self.tags)
        root[0] = 1.
        root = np.asarray(root, 'f')
        res = [[root]]
        for line in open(filepath, "r"):
            line = line.strip()
            if line == "":
                res[-1] = np.asarray(res[-1])
                res.append([root])
                continue
            items = [0.] + [float(v) for v in line.split("\t")[1:]]
            v = np.asarray(items, 'f')
            # zero out v[tag] for all tags not in top k
            ind = np.argpartition(v, -top)[-top:]
            mask = np.asarray([float(i in ind) for i in range(len(v))], np.float32)
            res[-1].append(v * mask)
        return res

    def assign_probs(self, sents, filepath):
        print >> sys.stderr, "assign_probs", filepath,
        probs = self._read_probs(filepath)
        for prob, sent in zip(probs, sents):
            sent.ts = prob
        print >> sys.stderr, "done"

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            return pickle.load(f)

class Sentence(object):
    def __init__(self, words, tags, labels, heads, ws, ts, ls):
        self.words  = words
        self.tags   = tags
        self.labels = labels
        self.heads  = heads
        self.ws = np.asarray(ws, 'i')
        self.ts = np.asarray(ts, 'i')
        self.ls = np.asarray(ls, 'i')
        self.orig_heads = None

    def __str__(self):
        res = ""
        for i in range(1, len(self)):
            word  = self.words[i]
            tag   = self.tags[i]
            head  = self.heads[i]
            label = self.labels[i]
            res += "{}\t{}\t{}\t{}\t-\t-\t{}\t{}\t-\t-\n".format(
                    i, word, tag, tag, head, label)
        return res

    def __len__(self):
        return len(self.ws)

    def projectivize(self):
        """
        projectivize input tree if it is non-projectivize
        find a deepest arc involved in the non-projectivity
        and attach the child to the head of the original head.
        if the resulting tree is projective the process ends, while not,
        go on to look for another candidate arc to projectivize.
        tokens: list of Token object
        output: bool
        """
        self.orig_heads = self.heads[:]
        ntokens = len(self)
        while True:
            left  = [-1] * ntokens
            right = [ntokens] * ntokens

            for i, head in enumerate(self.heads):
                l = min(i, head)
                r = max(i, head)

                for j in range(l+1, r):
                    if left[j] < l: left[j] = l
                    if right[j] > r: right[j] = r

            deepest_arc = -1
            max_depth = 0
            for i, head in enumerate(self.heads):
                if head == 0: continue
                l = min(i, head)
                r = max(i, head)
                lbound = max(left[l], left[r])
                rbound = min(right[l], right[r])

                if l < lbound or r > rbound:
                    depth = 0
                    j = i
                    while j != 0:
                        depth += 1
                        j = self.heads[j]
                    if depth > max_depth:
                        deepest_arc = i
                        max_depth = depth

            if deepest_arc == -1: return True
            lifted_head = self.heads[self.heads[deepest_arc]]
            self.heads[deepest_arc] = lifted_head


#########################################################
################### Transition System ###################
#########################################################

class System(object):
    NOOP    = 0
    SHIFT   = 1
    REDUCEL = 2
    REDUCER = 3

    def __init__(self, top, right, left, lchild, rchild,
                            lsibl, rsibl, sent, prev, prevact):
        self.top     = top
        self.right   = right
        self.left    = left
        self.lchild  = lchild
        self.rchild  = rchild
        self.lsibl   = lsibl
        self.rsibl   = rsibl
        self.sent    = sent
        self.prev    = prev
        self.prevact = prevact

    def isvalid(self, act):
        act = System.acttype(act)
        if act == System.NOOP:
            return False
        elif act == System.SHIFT:
            return self.has_input
        elif act == System.REDUCEL:
            return self.reducible and self.left.top != 0
        elif act == System.REDUCER:
            return self.reducible
        else:
            raise Exception

    def do_shift(self, act):
        n = System._null(self.sent)
        l = Vocab.unk_id
        return System(self.right, self.right + 1, self,
            (n, l), (n, l), n, n, self.sent, self, act)

    def do_reducel(self, act):
        l = System.act2label(act)
        left = self.left
        return System(self.top, self.right, left.left,
            (left, l), self.rchild, self, self.rsibl, self.sent, self, act)

    def do_reducer(self, act):
        l = System.act2label(act)
        left = self.left
        return System(left.top, self.right, left.left,
                left.lchild, (self, l), left.lsibl, left, self.sent, self, act)

    def expand(self, act):
        atype = System.acttype(act)
        if atype == System.SHIFT:
            return self.do_shift(act)
        elif atype == System.REDUCEL:
            return self.do_reducel(act)
        elif atype == System.REDUCER:
            return self.do_reducer(act)
        else:
            raise Exception()

    def expand_gold(self):
        # if self.isfinal: return []
        if not self.reducible: return self.expand(System.SHIFT)
        sent = self.sent
        s0h = sent.heads[self.top]
        s1h = sent.heads[self.left.top]
        if s1h == self.top:
            label = sent.ls[self.left.top]
            return self.expand(System.reducel(label))
        elif s0h == self.left.top:
            if all(h != self.top for h in sent.heads[self.right:]):
                label = sent.ls[self.top]
                return self.expand(System.reducer(label))
        return self.expand(System.SHIFT)

    def get_heads(s):
        res = [-1] * len(s.sent)
        while not s.prev.isnull:
            if not s.lchild[0].isnull: res[s.lchild[0].top] = s.top
            if not s.rchild[0].isnull: res[s.rchild[0].top] = s.top
            s = s.prev
        return res

    def get_labels(s):
        res = [0] * len(s.sent)
        while not s.prev.isnull:
            atype = System.acttype(s.prevact)
            if atype == System.REDUCEL:
                res[s.lchild[0].top] = s.lchild[1]
            elif atype == System.REDUCER:
                res[s.rchild[0].top] = s.rchild[1]
            s = s.prev
        return res

    @property
    def isnull(self):
        return self.top == 0 and self.right == 0

    @property
    def has_input(self):
        return self.right < len(self.sent)

    @property
    def reducible(self):
        return not self.left.isnull

    @property
    def isfinal(self):
        return not self.has_input and not self.reducible

    @staticmethod
    def reducel(label):
        return label << 1

    @staticmethod
    def reducer(label):
        return (label << 1) | 1

    @staticmethod
    def acttype(act):
        if act == 0:
            return System.NOOP
        elif act == 1:
            return System.SHIFT
        else:
            return 2 + (act & 1)

    @staticmethod
    def act2label(act):
        return act >> 1

    @staticmethod
    def gen(sent):
        null = System._null(sent)
        return System(0, 1, null, (null, Vocab.unk_id), (null, Vocab.unk_id),
                null, null, sent, null, System.NOOP)

    @staticmethod
    def _null(sent):
        s = System(0, 0, None, None, None, None,
                None, sent, None, System.NOOP)
        s.lchild, s.rchild = (s, Vocab.unk_id), (s, Vocab.unk_id)
        s.left, s.lsibl, s.rsibl = s, s, s
        return s

    def __str__(self):
        sent = self.sent
        stack = []
        s = self
        while not s.isnull:
            stack.append(
                sent.words[self.top] + "/" + sent.tags[self.top])
            s = s.left
        stack.reverse()
        buf = [w + "/" + t for w, t in
                zip(sent.words[self.right:], sent.tags[self.right:])]
        return "[" + " ".join(stack) + "] [" + " ".join(buf) + "]"

    def sparse_features(self):
        def idx_or_zero(idx):
            return idx if idx < len(self.sent) else 0

        idx = [idx_or_zero(self.right),          # b0
              idx_or_zero(self.right + 1),       # b1
              idx_or_zero(self.right + 2),       # b2
              idx_or_zero(self.right + 3),       # b3
              self.top,                          # s0
              self.lchild[0].top,                # s0l
              self.lsibl.lchild[0].top,          # s0l2
              self.rchild[0].top,                # s0r
              self.rsibl.rchild[0].top,          # s0r2
              self.lchild[0].lchild[0].top,      # s02l
              self.rchild[0].rchild[0].top,      # s12r
              self.left.top,                     # s1
              self.left.lchild[0].top,           # s1l
              self.left.lsibl.lchild[0].top,     # s1l2
              self.left.rchild[0].top,           # s1r
              self.left.rsibl.rchild[0].top,     # s1r2
              self.left.lchild[0].lchild[0].top, # s12l
              self.left.rchild[0].rchild[0].top, # s12r
              self.left.left.top,                # s2
              self.left.left.left.top]           # s3

        labels =  np.asarray(
                 [self.rchild[1],                # s0rc
                  self.rsibl.rchild[1],          # s0rc2
                  self.lchild[1],                # s0lc
                  self.lsibl.lchild[1],          # s0lc2
                  self.lchild[0].lchild[1],      # s02l
                  self.rchild[0].rchild[1],      # s02r
                  self.left.rchild[1],           # s1rc
                  self.left.rsibl.rchild[1],     # s1rc2
                  self.left.lchild[1],           # s1lc
                  self.left.lsibl.lchild[1],     # s1lc2
                  self.left.lchild[0].lchild[1], # s12l
                  self.left.rchild[0].rchild[1]  # s12r
                  ], 'i')

        words = self.sent.ws[idx]
        tags = self.sent.ts[idx]
        return words, tags, labels

#########################################################
#################### Neural Netwrork ####################
#########################################################

class Example(object):
    def __init__(self, w, t, l, valid, target):
        self.w      = w
        self.t      = t
        self.l      = l
        self.valid  = valid
        self.target = target

    @staticmethod
    def gen_train(st, targetsize, gpu=False):
        res = []
        while not st.isfinal:
            w, t, l = st.sparse_features()
            valid = np.asarray([float(st.isvalid(i))
                for i in range(targetsize)], 'f')
            st = st.expand_gold()
            target = np.asarray([st.prevact], 'i')

            if gpu:
                w = cuda.to_gpu(w)
                t = cuda.to_gpu(t)
                l = cuda.to_gpu(l)
                valid = cuda.to_gpu(valid)
                target = cuda.to_gpu(target)

            ex = Example(w, t, l, valid, target)
            res.append(ex)
        return res

    @staticmethod
    def gen_test(st, targetsize, gpu=False):
        w, t, l = st.sparse_features()
        valid = np.asarray([float(st.isvalid(i))
            for i in range(targetsize)], 'f')

        if gpu:
            w = cuda.to_gpu(w)
            t = cuda.to_gpu(t)
            l = cuda.to_gpu(l)
            valid = cuda.to_gpu(valid)

        return Example(w, t, l, valid, -1)

# def cubic(var):
#     return var ** 3

class FeedForward(Chain):
    def __init__(self, vocab, embedsize=50, hiddensize=1024, use_topk_tags=False,
            token_context_size=20, label_context_size=12, rescale_embed=True,
            wscale=1.):
        self.wordsize    = len(vocab.words)
        self.tagsize     = len(vocab.tags)
        self.labelsize   = len(vocab.labels)
        self.targetsize  = vocab.targetsize()
        self.embedsize   = embedsize
        self.contextsize = embedsize * (token_context_size * 2 + label_context_size)
        self.use_topk_tags = use_topk_tags

        super(FeedForward, self).__init__(
                w_embed = L.EmbedID(self.wordsize, embedsize),
                t_embed =
                (L.Linear if use_topk_tags else L.EmbedID)(self.tagsize, embedsize),
                l_embed = L.EmbedID(self.labelsize, embedsize),
                linear1 = L.Linear(self.contextsize, hiddensize, wscale=wscale),
                linear2 = L.Linear(hiddensize, self.targetsize, wscale=wscale)
                )

        # to ensure most ReLU units to activate in the first epochs
        self.linear1.b.data[:] = .2
        self.linear2.b.data[:] = .2

        if vocab.pretrained is not None:
            self.w_embed.W = Variable(vocab.pretrained)

        if rescale_embed:
            rescale(self.w_embed.W.data, .0, 1.)
            rescale(self.t_embed.W.data, .0, 1.)
            rescale(self.l_embed.W.data, .0, 1.)

        print >> sys.stderr, "(mean={:0.4f}, std={:0.4f})".format(
            np.mean(self.linear1.W.data), np.std(self.linear1.W.data))
        print >> sys.stderr, "(mean={:0.4f}, std={:0.4f})".format(
            np.mean(self.linear2.W.data), np.std(self.linear2.W.data))

        self.train = True
        self.drop_rate = .5

        print >> sys.stderr, "hiddensize =", hiddensize

    def set_train(self, train):
        self.train = train

    def __call__(self, batch):
        word_ids  = []
        tag_ids   = []
        label_ids = []
        valids    = []
        for ex in batch:
            word_ids.append(ex.w)
            tag_ids.append(ex.t)
            label_ids.append(ex.l)
            valids.append(ex.valid)

        word_ids = xp.concatenate(word_ids).reshape((len(batch), -1))
        tag_ids  = xp.concatenate(tag_ids).reshape((len(batch), -1))
        label_ids = xp.concatenate(label_ids).reshape((len(batch), -1))
        valids = xp.concatenate(valids).reshape((len(batch), -1))
        # # tag_ids = np.concatenate(tag_ids)

        # batch x token x embedsize
        h_w = self.w_embed(word_ids)
        h_t = self.t_embed(tag_ids)
        # h_t = F.reshape(h_t, (-1, 20, self.embedsize))
        h_l = self.l_embed(label_ids)

        # batch x [w; t; l] x embedsize
        h  = F.concat([h_w, h_t, h_l], 1)
        h1 = F.relu(self.linear1(h))
        h2 = F.dropout(h1, ratio=self.drop_rate, train=self.train)
        h3 = self.linear2(h2)
        return h3 * valids

    def predict(self, states):
        batch = map(lambda s: Example.gen_test(s, self.targetsize), states)
        logits = self(batch)
        return np.argmax(logits.data, 1)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            return pickle.load(f)

class WeightAveragedFF(FeedForward):
    def __init__(self, vocab, decay=0.99, **args):
        super(WeightAveragedFF, self).__init__(vocab, **args)
        self.decay = decay
        self.avg = FeedForward(vocab, **args)
        self.setup()

    def predict(self, states):
        return self.avg.predict(states)

    def setup(self):
        self.ave_params = self.__dict__["_children"]
        for param in self.ave_params:
            p = getattr(self, param)
            q = getattr(self.avg, param)

            if type(p) == L.EmbedID:
                print param
                q.W.data = p.W.data.copy()
            elif type(p) == L.Linear:
                print param
                q.W.data = p.W.data.copy()
                q.b.data = p.b.data.copy()

    def update_averaged(self, step):
        for param in self.ave_params:
            alpha = min(self.decay, (1. + step) / (10. + step))
            p = getattr(self, param)
            q = getattr(self.avg, param)
            q.W.data = alpha * q.W.data + (1 - alpha) * p.W.data
            if type(p) == L.Linear:
                q.b.data = alpha * q.b.data + (1 - alpha) * p.b.data

class ExponentialMovingAverage(object):
    # no support for GPU
    def __init__(self, initial_lr=.05, decay=0.96, decay_steps=4000):
        self.initial_lr  = initial_lr
        self.decay       = decay
        self.decay_steps = float(decay_steps)
        self.step = 1

    def __call__(self, opt):
        # lr = initial_lr * 0.96 (iter / decay_steps)
        opt.lr = self.initial_lr * self.decay ** (self.step / self.decay_steps)
        self.step += 1


def rescale(embed, rmean, rstd):
    print >> sys.stderr, "scaling embedding"
    mean = np.mean(embed)
    std  = np.std(embed)
    print >> sys.stderr, "(mean={:0.4f}, std={:0.4f}) ->".format(
            mean, std),
    embed = (embed - mean) * rstd / std + rmean
    print >> sys.stderr, "(mean={:0.4f}, std={:0.4f})".format(
            np.mean(embed), np.std(embed))

#########################################################
######################## Training #######################
#########################################################

class Parser(object):
    def __init__(self, vocab, model=FeedForward, batchsize=10000,
            niters=20000, do_averaging=True, evaliter=200, gpu=False, **args):
        self.vocab        = vocab
        self.batchsize    = batchsize
        self.niters       = niters
        self.evaliter     = evaliter
        self.model        = model(vocab, **args)
        self.targetsize   = vocab.targetsize()
        self.gpu          = gpu
        print >> sys.stderr, "gpu =", gpu

    def __call__(self, sents):
        """
        parse a batch of sentences
        TODO: change to return object representing
        parsed tree (e.g. list of Token) not System
        sents: list of list of Token
        output: list of System
        """
        res = []
        self.model.set_train(False)
        for i in range(0, len(sents), self.batchsize):
            batch = map(lambda s: System.gen(s), sents[i:i+self.batchsize])
            while not all(s.isfinal for s in batch):
                pred = self.model.predict(batch)
                for j in range(len(batch)):
                    if batch[j].isfinal: continue
                    batch[j] = batch[j].expand(pred[j])
            res.extend(batch)
        self.model.set_train(True)
        return res

    def gen_trainexamples(self, sents):
        res = []
        print >> sys.stderr, "creating training examples..."
        for i, sent in enumerate(sents):
            sys.stderr.write("\r{}/{}".format(i, len(sents)))
            sys.stderr.flush()
            s = System.gen(sent)
            res.extend(Example.gen_train(s, self.targetsize, self.gpu))
        print >> sys.stderr, "\ndone"
        return res

    def train(self, trainsents, testsents, parserfile):
        classifier = L.Classifier(self.model)

        if self.gpu:
            cuda.get_device().use()
            classifier = classifier.to_gpu()

        trainexamples = self.gen_trainexamples(trainsents)

        optimizer = O.AdaGrad(.01, 1e-6)
        optimizer.setup(classifier)
        optimizer.add_hook(WeightDecay(1e-8))
        # optimizer = O.MomentumSGD(.05, .9)
        # optimizer.setup(classifier)
        # optimizer.add_hook(WeightDecay(1e-4))
        # optimizer.add_hook(ExponentialMovingAverage())

        best_uas = 0.
        print >> sys.stderr, "will run {} iterations".format(self.niters)
        for step in range(1, self.niters+1):
            batch = random.sample(trainexamples, self.batchsize)
            t = Variable(xp.concatenate(map(lambda ex: ex.target, batch)))
            optimizer.update(classifier, batch, t)
            if type(self.model) == WeightAveragedFF:
                self.model.update_averaged(step)

            print >> sys.stderr, "Epoch:{}\tloss:{}\tacc:{}".format(
                    step, classifier.loss.data, classifier.accuracy.data)

            if step % self.evaliter == 0:
                print >> sys.stderr, "Evaluating model on dev data..."
                res = self(testsents)
                uas, las = accuracy(res)

                if uas > best_uas:
                    print >> sys.stderr, "Best score. Saving parser...",
                    self.save(parserfile)
                    best_uas = uas
                    print >> sys.stderr, "done"

        self.save(parserfile + ".final")
        print >> sys.stderr, "done"

    def save(self, path):
        with open(path, "wb") as f:
            if self.gpu:
                self.model.to_cpu()
                pickle.dump(self, f)
                self.model.to_gpu()
            else:
                pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            return pickle.load(f)

def accuracy(states, verbose=True):
    ignore = ["''", ",", ".", ":", "``"]
    unum, lnum, den = 0, 0, 0
    for s in states:
        predheads  = s.get_heads()
        predlabels = s.get_labels()
        goldheads  = s.sent.heads
        goldlabels = s.sent.ls
        for i, word in enumerate(s.sent.words):
            if i == 0 or word in ignore: continue # skip root
            den += 1
            if predheads[i] == goldheads[i]:
                unum += 1
                if predlabels[i] == goldlabels[i]:
                    lnum += 1
    uas = float(unum) / float(den)
    las = float(lnum) / float(den)
    if verbose: print >> sys.stderr, "UAS:{:0.4f}\tLAS:{:0.4f}".format(uas, las)
    return uas, las

#########################################################
########################## Main #########################
#########################################################

def main():
    word_path       = "chen/words.lst"
    tag_path        = "../jackknife/data/tags.lst"
    label_path      = "labels.txt"
    embed_path      = "chen/embeddings.txt"
    train_path      = "corpus/wsj_02-21.sd.orig.tagged"
    train_prob_path = "../jackknife/wsj_02-21.probs"
    test_path       = "corpus/wsj_23.sd.orig.tagged"
    test_prob_path  = "../jackknife/wsj_23.probs"
    out_path        = "parser_syntaxnet.dat"
    vocab      = Vocab(word_path, tag_path, label_path, embed_path)
    trainsents = vocab.read_conll_train(train_path)
    # vocab.assign_probs(trainsents, train_prob_path)
    for sent in trainsents:
        sent.projectivize()
    testsents  = vocab.read_conll_test(test_path)
    # vocab.assign_probs(testsents, test_prob_path)
    parser     = Parser(vocab, model=FeedForward, gpu=USE_GPU, niters=40000, evaliter=4000)
    # parser     = Parser(vocab, model=WeightAveragedFF, gpu=False, niters=30000,
    #                     evaliter=400, hiddensize=2048, rescale_embed=False, wscale=0.1)
    parser.train(trainsents, testsents, out_path)
    res        = parser(testsents)
    uas, las   = accuracy(res)

if __name__ == '__main__':
    main()
