#!/usr/bin/python
from collections import defaultdict
from chainer import Chain, Variable
import chainer.functions as F
import chainer.optimizers as O
from chainer.optimizer import WeightDecay
# from chainer import cuda
import chainer.links as L
import numpy as np
import random
import sys
# from tqdm import tqdm
import cPickle as pickle

#########################################################
################ Vocabulary, Token, Tree ################
#########################################################

class Vocab(object):
    pad = "PADDING"
    unk = "UNKNOWN"

    def __init__(self, word_path, tag_path=None, cutoff=1, pretrained_path=None):
        self.words  = Vocab._read_listfile(word_path)
        self.labels = defaultdict(lambda: len(self.tags))
        self.labels[Vocab.unk] = 0
        self.freq   = None
        self.cutoff = cutoff

        if tag_path is not None:
            self.tags   = Vocab._read_listfile(tag_path)
        else:
            self.tags   = defaultdict(lambda: len(self.tags))
            self.tags[Vocab.unk] = 0

        if pretrained_path is not None:
            self.pretrained = Vocab._read_pretrained(pretrained_path)
        else:
            self.pretrained = None

    @staticmethod
    def _freqdict(path):
        res = defaultdict(lambda: 0)
        for line in open(path):
            line = line.strip()
            if len(line) == 0: continue
            items = line.split("\t")
            word = items[1]
            res[Vocab.normalize(word)] += 1
        return res

    def _accept_new_entries(self, acc):
        if acc:
            self.tags.default_factory = Vocab._get_id(self.tags)
            self.labels.default_factory = Vocab._get_id(self.labels)
        else:
            self.tags.default_factory = Vocab._get_unk(self.tags)
            self.labels.default_factory = Vocab._get_unk(self.labels)

    @staticmethod
    def _read_listfile(path):
        res = defaultdict(lambda: res[Vocab.unk])
        res[Vocab.unk] = 0
        for i, line in enumerate(open(path)):
            res[line.strip()] = i + 1
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

    def create_token(self, wstr, tstr, head, lstr):
        norm_wstr = Vocab.normalize(wstr)
        map_unk   = self.freq[norm_wstr] < self.cutoff
        word      = self.words[Vocab.unk] if map_unk else self.words[norm_wstr]
        tag       = self.tags[tstr]
        label     = self.labels[lstr]
        return Token(word, tag, head, label, wstr, tstr, lstr)

    def roottoken(self):
        return self.create_token(Vocab.pad, Vocab.unk, -1, Vocab.unk)

    def targetsize(self):
        # NOOP: 0
        # SHIFT: 1
        # REDUCEL(i) >=2 even, i in [1 to labelsize]
        # REDUCER(i) >=3 odd, i in [1 to labelsize]
        # REDUCE(0) == REDUCE(UNKNOWN) is ignored
        nlabels = len(self.labels) - 1
        return 2 + 2 * nlabels

    def read_conll_train(self, filepath):
        self._accept_new_entries(True)
        if self.freq is None:
            self.freq = self._freqdict(filepath)
        return self._read_conll(filepath)

    def read_conll_test(self, filepath):
        self._accept_new_entries(False)
        res = self._read_conll(filepath)
        self._accept_new_entries(True)
        return res

    def _read_conll(self, filepath):
        res = [[self.roottoken()]]
        for line in open(filepath, "r"):
            line = line.strip()
            if line == "":
                res.append([self.roottoken()])
                continue
            items = line.split("\t")
            word, tag, label = items[1], items[4], items[7]
            head = int(items[6])
            t = self.create_token(word, tag, head, label)
            res[-1].append(t)
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
        print >> sys.stderr, "reading prob file:", filepath,
        root = [.0]*len(self.tags)
        root[0] = 1.
        root = np.asarray(root, dtype=np.float32)
        res = [[root]]
        for line in open(filepath, "r"):
            line = line.strip()
            if line == "":
                res.append([root])
                continue
            items = [0.] + [float(v) for v in line.split("\t")[1:]]
            v = np.asarray(items, dtype=np.float32)
            ind = np.argpartition(v, -top)[-top:]
            mask = np.asarray([float(i in ind) for i in range(len(v))], np.float32)
            res[-1].append(v * mask)
        print >> sys.stderr, "done"
        return res

    def assign_probs(self, sents, filepath):
        print >> sys.stderr, "assign_probs", filepath,
        probs = self._read_probs(filepath)
        for prob, sent in zip(probs, sents):
            for p, t in zip(prob, sent):
                t.tag = p
        print >> sys.stderr, "done"

    @staticmethod
    def _get_unk(d):
        return lambda: d[Vocab.unk]

    @staticmethod
    def _get_id(d):
        return lambda: len(d)

    def setup_save(self):
        self.words.default_factory  = None
        self.tags.default_factory   = None
        self.labels.default_factory = None
        if self.freq is not None:
            self.freq.default_factory = None

    def _load(self):
        self.words.default_factory  = Vocab._get_id(self.words)
        self.tags.default_factory   = Vocab._get_id(self.tags)
        self.labels.default_factory = Vocab._get_id(self.labels)
        if self.freq is not None:
            self.freq.default_factory = Vocab._get_id(self.freq)

    def save(self, path):
        with open(path, "wb") as f:
            self.setup_save()
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            res = pickle.load(f)
            res._load()
            return res

class Token(object):
    def __init__(self, word, tag, head, label, wstr, tstr, lstr):
        self.word  = word
        self.tag   = tag
        self.head  = head
        self.label = label
        self.wstr  = wstr
        self.tstr  = tstr
        self.lstr  = lstr

    def conll(self):
        items = [self.wstr, self.tstr, self.tstr, "-",
                "-", str(self.head), self.lstr, "-", "-"]
        return "\t".join(items)

    @staticmethod
    def sent2conll(tokens):
        for i, t in enumerate(tokens):
            if i == 0: continue
            print str(i) + "\t" + t.conll()

def projectivize(tokens):
    """
    projectivize input tree if it is non-projectivize
    find a deepest arc involved in the non-projectivity
    and attach the child to the head of the original head.
    if the resulting tree is projective the process ends, while not,
    go on to look for another candidate arc to projectivize.
    tokens: list of Token object
    output: bool
    """
    ntokens = len(tokens)
    while True:
        left  = [-1] * ntokens
        right = [ntokens] * ntokens

        for i in range(ntokens):
            head = tokens[i].head
            l = min(i, head)
            r = max(i, head)

            for j in range(l+1, r):
                if left[j] < l: left[j] = l
                if right[j] > r: right[j] = r

        deepest_arc = -1
        max_depth = 0
        for i in range(ntokens):
            head = tokens[i].head
            if head == 0: continue
            l = min(i, head)
            r = max(i, head)
            lbound = max(left[l], left[r])
            rbound = min(right[l], right[r])

            if l < lbound or r > rbound:
                depth = 0
                j = i
                while j != -1:
                    depth += 1
                    j = tokens[j].head
                if depth > max_depth:
                    deepest_arc = i
                    max_depth = depth

        if deepest_arc == -1: return True
        lifted_head = tokens[tokens[deepest_arc].head].head
        tokens[deepest_arc].head = lifted_head


#########################################################
################### Transition System ###################
#########################################################

class System(object):
    NOOP    = 0
    SHIFT   = 1
    REDUCEL = 2
    REDUCER = 3

    def __init__(self, step, score, top, right, left, lchild, rchild,
            lsibl, rsibl, tokens, prev, prevact):
        self.step    = step
        self.score   = score
        self.top     = top
        self.right   = right
        self.left    = left
        self.lchild  = lchild
        self.rchild  = rchild
        self.lsibl   = lsibl
        self.rsibl   = rsibl
        self.tokens  = tokens
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

    def expand(self, act):
        atype = System.acttype(act)
        if atype == System.SHIFT:
            null = System._null(self.tokens)
            return System(self.step + 1, 0.0, self.right, self.right + 1,
                    self, null, null, null, null, self.tokens, self, act)
        elif atype == System.REDUCEL:
            left = self.left
            return System(self.step + 1, 0.0, self.top, self.right, left.left,
                    left, self.rchild, self, self.rsibl, self.tokens, self, act)
        elif atype == System.REDUCER:
            left = self.left
            return System(self.step + 1, 0.0, left.top, self.right, left.left,
                    left.lchild, self, left.lsibl, self.left, self.tokens, self, act)
        else:
            raise Exception()

    # def expand_pred(self):
    #     if self.isfinal: return []
    #     res = []
    #     if self.reducible:
    #         for label in range(1, System.nlabels):
    #             res.append(self.expand(System.reducer(label)))
    #         if self.left.top != 0:
    #             for label in range(1, System.nlabels):
    #                 res.append(self.expand(System.reducel(label)))
    #     if self.has_input: res.append(self.expand(System.SHIFT))
    #     return res

    def expand_gold(self):
        # if self.isfinal: return []
        if not self.reducible: return self.expand(System.SHIFT)
        s0 = self.top_token()
        s1 = self.left.top_token()
        if s1.head == self.top:
            label = s1.label
            return self.expand(System.reducel(label))
        elif s0.head == self.left.top:
            if all(t.head != self.top for t in self.tokens[self.right:]):
                label = s0.label
                return self.expand(System.reducer(label))
        return self.expand(System.SHIFT)

    def label_at(self, child):
        s = self
        while s.step != child.step + 1:
            if s.prev.isnull: break
            s = s.prev
        return System.act2label(s.prevact)

    def get_heads(self):
        res = [-1] * len(self.tokens)
        s = self
        while not s.prev.isnull:
            if not s.lchild.isnull: res[s.lchild.top] = s.top
            if not s.rchild.isnull: res[s.rchild.top] = s.top
            s = s.prev
        return res

    def get_labels(self):
        res = [0] * len(self.tokens)
        s = self
        while not s.prev.isnull:
            atype = System.acttype(s.prevact)
            if atype == System.REDUCEL:
                res[s.lchild.top] = System.act2label(s.prevact)
            elif atype == System.REDUCER:
                res[s.rchild.top] = System.act2label(s.prevact)
            s = s.prev
        return res

    def top_token(self):
        return self.tokens[self.top]

    @property
    def isnull(self):
        return self.step == 0

    @property
    def nlabels(self):
        raise Exception("not implemented")

    @property
    def has_input(self):
        return self.right < len(self.tokens)

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
    def gen(tokens):
        null = System._null(tokens)
        return System(1, 0.0, 0, 1, null, null, null,
            null, null, tokens, null, System.NOOP)

    @staticmethod
    def _null(tokens):
        s = System(0, 0.0, 0, 0, None, None, None, None,
                None, tokens, None, System.NOOP)
        s.left    = s
        s.lchild  = s
        s.rchild  = s
        s.lsibl   = s
        s.rsibl   = s
        return s

    def __str__(self):
        stack = []
        s = self
        while not s.isnull:
            t = self.tokens[s.top]
            stack.append(t.wstr + "/" + t.tstr)
            s = s.left
        stack.reverse()
        buf = [t.wstr + "/" + t.tstr for t in self.tokens[self.right:]]
        return "[" + " ".join(stack) + "] [" + " ".join(buf) + "]"

    def tokenat(self, i):
        if i < len(self.tokens):
            return self.tokens[i]
        else:
            return self.tokens[0]

    def sparse_features(self):
        b0 = self.tokenat(self.right)
        b1 = self.tokenat(self.right + 1)
        b2 = self.tokenat(self.right + 2)
        b3 = self.tokenat(self.right + 3)
        s0 = self.top_token()
        s0l  = self.lchild.top_token()
        s0l2 = self.lsibl.lchild.top_token()
        s0r  = self.rchild.top_token()
        s0r2 = self.rsibl.rchild.top_token()
        s02l = self.lchild.lchild.top_token()
        s12r = self.rchild.rchild.top_token()
        s1   = self.left.top_token()
        s1l  = self.left.lchild.top_token()
        s1l2 = self.left.lsibl.lchild.top_token()
        s1r  = self.left.rchild.top_token()
        s1r2 = self.left.rsibl.rchild.top_token()
        s12l = self.left.lchild.lchild.top_token()
        s12r = self.left.rchild.rchild.top_token()
        s2   = self.left.left.top_token()
        s3   = self.left.left.left.top_token()

        s0rc_label  = self.label_at(self.rchild)
        s0rc2_label = self.label_at(self.rsibl.rchild)
        s0lc_label  = self.label_at(self.lsibl)
        s0lc2_label = self.label_at(self.lsibl.lsibl)
        s02l_label  = self.label_at(self.lsibl.left.lsibl)
        s02r_label  = self.label_at(self.rchild.rchild)
        s1rc_label  = self.label_at(self.left.rchild)
        s1rc2_label = self.label_at(self.left.rsibl.rchild)
        s1lc_label  = self.label_at(self.left.lsibl)
        s1lc2_label = self.label_at(self.left.lsibl.lsibl)
        s12l_label  = self.label_at(self.left.lsibl.left.lsibl)
        s12r_label  = self.label_at(self.left.rchild.rchild)

        words = np.asarray(
                [b0.word, b1.word, b2.word, b3.word, s0.word, s0l.word, s0l2.word,
        s0r.word, s0r2.word, s02l.word, s12r.word, s1.word, s1l.word, s1l2.word,
        s1r.word, s1r2.word, s12l.word, s12r.word, s2.word, s3.word],
                dtype=np.int32)

        tags = np.asarray(
                [b0.tag, b1.tag, b2.tag, b3.tag, s0.tag, s0l.tag, s0l2.tag,
        s0r.tag, s0r2.tag, s02l.tag, s12r.tag, s1.tag, s1l.tag, s1l2.tag,
        s1r.tag, s1r2.tag, s12l.tag, s12r.tag, s2.tag, s3.tag],
                dtype=np.float32)
                # dtype=np.int32)

        labels = np.asarray(
                [s0rc_label, s0rc2_label, s0lc_label, s0lc2_label, s02l_label,
        s02r_label, s1rc_label, s1rc2_label, s1lc_label, s1lc2_label,
        s12l_label, s12r_label],
                dtype=np.int32)

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
                for i in range(targetsize)], dtype=np.float32)
            st = st.expand_gold()
            target = np.asarray([st.prevact], dtype=np.int32)

            # if gpu:
            #     w = cuda.to_gpu(w)
            #     t = cuda.to_gpu(t)
            #     l = cuda.to_gpu(l)
            #     valid = cuda.to_gpu(valid)
            #     target = cuda.to_gpu(target)

            ex = Example(w, t, l, valid, target)
            res.append(ex)
        return res

    @staticmethod
    def gen_test(st, targetsize):
        w, t, l = st.sparse_features()
        valid = np.asarray([float(st.isvalid(i))
            for i in range(targetsize)], dtype=np.float32)
        return Example(w, t, l, valid, -1)

# def cubic(var):
#     return var ** 3

class FeedForward(Chain):
    def __init__(self, vocab, embedsize=50, hiddensize=1024, use_topk_tags=True,
            token_context_size=20, label_context_size=12, rescale_embed=True,
            wscale=1., averaging=True):
        self.wordsize    = len(vocab.words)
        self.tagsize     = len(vocab.tags)
        self.labelsize   = len(vocab.labels)
        self.targetsize  = vocab.targetsize()
        self.embedsize   = embedsize
        self.contextsize = embedsize * (token_context_size * 2 + label_context_size)
        self.use_topk_tags = use_topk_tags

        super(FeedForward, self).__init__(
                w_embed = L.EmbedID(self.wordsize, embedsize),
                t_embed = L.Linear(self.tagsize, embedsize),
                # t_embed = L.EmbedID(self.tagsize, embedsize),
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

        self.averaged = None
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

        # word_ids = cuda.cupy.concatenate(word_ids).reshape((-1, len(batch)))
        # tag_ids  = cuda.cupy.concatenate(tag_ids).reshape((-1, len(batch)))
        # label_ids = cuda.cupy.concatenate(label_ids).reshape((-1, len(batch)))
        # valids = cuda.cupy.concatenate(valids).reshape((-1, len(batch)))
        word_ids = np.asarray(word_ids)
        # tag_ids  = np.asarray(tag_ids)
        tag_ids = np.concatenate(tag_ids)
        label_ids = np.asarray(label_ids)
        valids = np.asarray(valids)

        # batch x token x embedsize
        h_w = self.w_embed(word_ids)
        h_t = self.t_embed(tag_ids)
        h_t = F.reshape(h_t, (-1, 20, self.embedsize))
        h_l = self.l_embed(label_ids)

        # batch x [w; t; l] x embedsize
        h  = F.concat([h_w, h_t, h_l], 1)
        h1 = F.relu(self.linear1(h))
        # h2 = F.dropout(h1, ratio=self.drop_rate, train=self.train)
        h2 = h1
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

class WeightAveraged(FeedForward):
    def __init__(self, vocab, decay=0.99, **args):
        super(WeightAveraged, self).__init__(vocab, rescale_embed=False, **args)
        self.decay = decay
        self.step = 1

    def setup(self, model):
        self.params = model.__dict__["_children"]
        for param in self.params:
            p = getattr(self, param)
            q = getattr(model, param)

            if type(p) == L.EmbedID:
                p.W.data = q.W.data.copy()
            elif type(p) == L.Linear:
                p.W.data = q.W.data.copy()
                p.b.data = q.b.data.copy()

        model.averaged = self
        self.parent = model

    def update(self):
        for param in self.params:
            alpha = min(self.decay, (1. + self.step) / (10. + self.step))
            p = getattr(self, param)
            q = getattr(self.parent, param)
            p.W.data = alpha * p.W.data + (1 - alpha) * q.W.data
            if type(p) == L.Linear:
                p.b.data = alpha * p.b.data + (1 - alpha) * q.b.data
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
    def __init__(self, vocab, batchsize=10000, niters=20000,
            do_averaging=True, evaliter=200, gpu=False, **args):
        self.vocab        = vocab
        self.batchsize    = batchsize
        self.niters       = niters
        self.evaliter     = evaliter
        self.model        = FeedForward(vocab, **args)
        self.targetsize   = vocab.targetsize()
        self.do_averaging = do_averaging
        self.gpu          = gpu
        print >> sys.stderr, "gpu =", gpu

    def __call__(self, sents, do_averaging):
        """
        parse a batch of sentences
        TODO: change to return object representing
        parsed tree (e.g. list of Token) not System
        sents: list of list of Token
        output: list of System
        """
        res = []
        model = self.model if not do_averaging else self.model.averaged
        model.set_train(False)
        for i in range(0, len(sents), self.batchsize):
            batch = map(lambda s: System.gen(s), sents[i:i+self.batchsize])
            while not all(s.isfinal for s in batch):
                pred = model.predict(batch)
                for j in range(len(batch)):
                    if batch[j].isfinal: continue
                    batch[j] = batch[j].expand(pred[j])
            res.extend(batch)
        model.set_train(True)
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
        trainexamples = self.gen_trainexamples(trainsents)

        classifier = L.Classifier(self.model)
        if self.do_averaging:
            WeightAveraged(self.vocab).setup(self.model)

        if self.gpu:
            cuda.get_device().use()
            classifier.to_gpu()

        # optimizer = O.AdaGrad(.01, 1e-6)
        # optimizer.setup(classifier)
        # optimizer.add_hook(WeightDecay(1e-8))
        optimizer = O.MomentumSGD(.05, .9)
        optimizer.setup(classifier)
        optimizer.add_hook(WeightDecay(1e-4))

        best_uas = 0.
        print >> sys.stderr, "will run {} iterations".format(self.niters)
        for i in range(1, self.niters+1):
            # lr = initial_lr * 0.96 (iter / decay_steps)
            lr = .05 * .96 ** (i / 4000.)
            optimizer.lr = lr
            batch = random.sample(trainexamples, self.batchsize)
            t = Variable(np.concatenate(map(lambda ex: ex.target, batch)))
            # t = Variable(cuda.cupy.concatenate(map(lambda ex: ex.target, batch)))
            optimizer.update(classifier, batch, t)
            self.model.averaged.update()

            print >> sys.stderr, "Epoch:{}\tloss:{}\tacc:{}".format(
                    i, classifier.loss.data, classifier.accuracy.data)

            if i % self.evaliter == 0:
                print >> sys.stderr, "Evaluating model on dev data..."
                print >> sys.stderr, "without averaging",
                res = self(testsents, False)
                uas, las = accuracy(res)

                print >> sys.stderr, "with averaging",
                res = self(testsents, True)
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
            self.vocab.setup_save()
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
        goldheads = map(lambda t: t.head, s.tokens)
        goldlabels = map(lambda t: t.label, s.tokens)
        for i in range(1, len(s.tokens)): # skip root
            if s.tokens[i].wstr in ignore: continue
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
    tag_path        = "../jackknife/data/tags.lst"
    word_path       = "chen/words.lst"
    embed_path      = "chen/embeddings.txt"
    train_path      = "corpus/wsj_02-21.sd.orig.tagged"
    train_prob_path = "../jackknife/wsj_02-21.probs"
    test_path       = "corpus/wsj_23.sd.orig.tagged"
    test_prob_path  = "../jackknife/wsj_23.probs"
    out_path        = "parser_syntaxnet.dat"
    vocab      = Vocab(word_path, tag_path,embed_path)
    trainsents = vocab.read_conll_train(train_path)
    vocab.assign_probs(trainsents, train_prob_path)
    for sent in trainsents:
        projectivize(sent)
    testsents  = vocab.read_conll_test(test_path)
    vocab.assign_probs(testsents, test_prob_path)
    parser     = Parser(vocab, gpu=False, niters=30000,
            hiddensize=2048, rescale_embed=False, wscale=0.1)
    parser.train(trainsents, testsents, out_path)
    res        = parser(testsents)
    uas, las   = accuracy(res)

if __name__ == '__main__':
    main()
