import asciitree
import numpy as np
import pickle
import sys
import argparse
from pathlib import Path
from collections import OrderedDict, deque
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, Iterator

import chainer
from chainer import Variable, training
from chainer.training import extensions
from chainer.optimizer import WeightDecay
import chainer.functions as F
import chainer.links as L

# try:
#     import cupy
#     Array = Union[np.ndarray, cupy.ndarray]
# except ImportError:
Array = np.ndarray
VariableOrArray = Union[Variable, Array]

FeatureSet = NamedTuple(
    'FeatureSet',
    [('words', VariableOrArray),
    ('tags', VariableOrArray),
    ('labels', VariableOrArray),
])

TrainingExample = NamedTuple(
    'TrainingExample',
    [('words', VariableOrArray),
    ('tags', VariableOrArray),
    ('labels', VariableOrArray),
    ('valid_actions', VariableOrArray),
    ('gold_action', VariableOrArray),
])

UNK = 'UNKNOWN'
PAD = 'PADDING'
UNK_ID = 0
PAD_ID = 1
IGNORE = -1


def log(msg):
    print('log:', msg, file = sys.stderr)


def normalize(word: str) -> str:
    n = word.lower()
    if n == "-lrb-":
        return "("
    elif n == "-rrb-":
        return ")"
    else:
        return n


def read_vocab(path: Path) -> Dict[str, int]:
    res = {UNK: UNK_ID}
    for line in path.open():
        entry = line.strip()
        # assert entry not in res, \
        #     ('failure is Dataloader.read_vocab: '
        #     f'duplicate entry: {entry}.')
        res[entry] = len(res)
    return res


def read_pretrained(path: Path) -> Array:
    log(f'loading pretrained embeddings from {path}')
    io = open(path)
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



class Sentence(object):
    def __init__(self,
            words:  List[str],
            lemmas: List[str],
            tags1:  List[str],
            tags2:  List[str],
            labels: List[str],
            heads:  List[int],
            word_ids:  Array,
            tag_ids:   Array,
            label_ids: Array,
            root_node = True) -> None:
        self.words = words
        self.lemmas = lemmas
        self.tags1 = tags1
        self.tags2 = tags2
        self.labels = labels
        self.heads = heads
        self.word_ids = word_ids
        self.tag_ids = tag_ids
        self.label_ids = label_ids
        self.root_node = root_node

    @property
    def start(self):
        return 1 if self.root_node else 0

    def __str__(self) -> str:
        """
            return string in CoNLL format.
        """
        res = ''
        for i in range(self.start, len(self)):
            items = [
                str(i),
                self.words[i],
                self.lemmas[i],
                self.tags1[i],
                self.tags2[i],
                '-',
                str(self.heads[i]),
                self.labels[i],
                '-', 
                '-' ]
            res += '\t'.join(items) + '\n'
        return res

    def __len__(self) -> int:
        return len(self.words)

    def visualize(self) -> str:
        token_str = ['root']
        children: List[List[int]] = [[] for _ in range(len(self))]
        for i in range(self.start, len(self)):
            token_str.append(f'{self.words[i]} @{i}')
            children[self.heads[i]].append(i)

        def to_dict(i: int) -> OrderedDict:
            d: OrderedDict = OrderedDict()
            for c in children[i]:
                d[token_str[c]] = to_dict(c)
            return d

        tree: OrderedDict = OrderedDict()
        tree[token_str[0]] = to_dict(0)
        res = asciitree.LeftAligned()(tree)
        return res

    def projectivize(self) -> None:
        """
            projectivize input tree if it is non-projectivize
            find a deepest arc involved in the non-projectivity
            and attach the child to the head of the original head.
            if the resulting tree is projective the process ends, while not,
            go on to look for another candidate arc to projectivize.
        """
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

            if deepest_arc == -1: return
            lifted_head = self.heads[self.heads[deepest_arc]]
            self.heads[deepest_arc] = lifted_head


class DataLoader(object):
    def __init__(self,
                word_path: Path,
                tag_path: Path,
                label_path: Path,
                pretrained_path: Optional[Path] = None) -> None:
        self.words = read_vocab(word_path)
        self.tags = read_vocab(tag_path)
        self.labels = read_vocab(label_path)
        self.pretrained = read_pretrained(pretrained_path) \
                if pretrained_path is not None else None

    def _get_or_add_entry(
            self, entry: str, accept_new_entry: bool) -> int:
        entry = normalize(entry)
        if entry in self.words:
            return self.words[entry]
        elif accept_new_entry:
            idx = len(self.words)
            self.words[entry] = idx
            return idx
        else:
            return UNK_ID

    @property
    def action_size(self) -> int:
        """
            NOOP -> 0
            SHIFT -> 1
            REDUCEL(action_id) -> some n >= 2 and n is even
            REDUCER(action_id) -> some n >= 2 and n is odd
            REDUCE(0) -> REDUCE(UNKNOWN) is ignored
        """
        nlabels = len(self.labels) - 1
        return 2 + 2 * nlabels

    def read_conll(
                self,
                filename: Path,
                accept_new_entry: bool = True) -> Iterator[Sentence]:
        """
        reads .conll file and returns a list of Sentence
        """
        def new_kwargs() -> Dict[str, List]:
            return dict(
            words = [PAD],
            lemmas = [PAD],
            tags1 = [UNK],
            tags2 = [UNK],
            labels = [UNK],
            heads = [0],
            word_ids = [PAD_ID],
            tag_ids = [UNK_ID],
            label_ids = [UNK_ID],
        )
        kwargs = new_kwargs()
        for line in filename.open():
            line = line.strip()
            if line == '':
                kwargs['word_ids'] = np.asarray(kwargs['word_ids'], int)
                kwargs['tag_ids'] = np.asarray(kwargs['tag_ids'], int)
                kwargs['label_ids'] = np.asarray(kwargs['label_ids'], int)
                yield Sentence(**kwargs)
                kwargs = new_kwargs()
                continue
            items = line.split('\t')
            kwargs['words'].append(items[1])
            kwargs['lemmas'].append(items[2])
            kwargs['tags1'].append(items[3])
            kwargs['tags2'].append(items[4])
            kwargs['labels'].append(items[7])
            kwargs['heads'].append(int(items[6]))
            kwargs['word_ids'].append(
                    self._get_or_add_entry(items[1], accept_new_entry))
            kwargs['tag_ids'].append(self.tags[items[3]])
            kwargs['label_ids'].append(self.labels[items[7]])

    def read_conll_test(self, filename: Path) -> List[Sentence]:
        sents = list(self.read_conll(filename, False))
        return sents

    def read_conll_train(self, filename: Path) -> List[Sentence]:
        """
            load CoNLL sentences, update vocabulary (`DataLoader.words`)
            and add entries for new words to the pretrained embedding matrix.
        """
        sents = list(self.read_conll(filename, True))
        if self.pretrained is not None:
            log('adding entries for new words to the pretrained embeddings')
            old_vocab_size, units = self.pretrained.shape
            new_pretrained = 0.02 * np.random.random_sample(
                                    (len(self.words), units)) - 0.01
            new_pretrained[:old_vocab_size, :] = self.pretrained
            self.pretrained = new_pretrained
            log(f'done. {old_vocab_size} ---> {len(self.words)}')
        return sents

    def save(self, filename: Path) -> None:
        with filename.open('wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: Path) -> 'DataLoader':
        with filename.open() as f:
            return pickle.load(f)



########################################################################
######################### Utility Class for Action #####################
########################################################################


class Action(object):
    NOOP   : int = 0
    SHIFT  : int = 1
    REDUCEL: int = 2
    REDUCER: int = 3

    @staticmethod
    def to_type(act: int) -> int:
        """
        turn action id to action type,
        `NOOP`, `SHIFT`, `REDUCEL` or `REDUCER`.
        """
        if act == 0:
            return Action.NOOP
        elif act == 1:
            return Action.SHIFT
        else:
            return 2 + (act & 1)

    @staticmethod
    def to_label(act: int) -> int:
        """
        turn `REDUCEL` or `REDUCER` to dependency label id.
        """
        return act >> 1

    @staticmethod
    def reducel(label: int) -> int:
        return label << 1

    @staticmethod
    def reducer(label: int) -> int:
        return (label << 1) | 1



########################################################################
########################## Transition System ###########################
########################################################################

LabeledState = NamedTuple(
    'LabeledState',
    [('state', 'State'),
    ('label', int)]
)

class State(object):
    def __init__(self,
                top: int,
                right: int,
                left: 'State',
                lchild1: LabeledState,
                rchild1: LabeledState,
                lchild2: LabeledState,
                rchild2: LabeledState,
                sent: Sentence, 
                prev: 'State',
                prevact: int) -> None:
        self.top     = top
        self.right   = right
        self.left    = left
        self.lchild1 = lchild1
        self.rchild1 = rchild1
        self.lchild2 = lchild2
        self.rchild2 = rchild2
        self.sent    = sent
        self.prev    = prev
        self.prevact = prevact

    def is_valid(self, act: int) -> bool:
        act_type = Action.to_type(act)
        if act_type == Action.NOOP:
            return False
        elif act_type == Action.SHIFT:
            return self.has_input
        elif act_type == Action.REDUCEL:
            return self.reducible and self.left.top != 0
        elif act_type == Action.REDUCER:
            return self.reducible
        else:
            raise RuntimeError(
                f'unexpected action id in State.is_valid: {act}')

    def shift(self, act: int) -> 'State':
        dummy = LabeledState(
            State._dummy(self.sent),
            UNK_ID)
        return State(
                top = self.right,
                right = self.right + 1,
                left = self,
                lchild1 = dummy,
                rchild1 = dummy,
                lchild2 = dummy,
                rchild2 = dummy,
                sent = self.sent,
                prev = self,
                prevact = act)

    def reducel(self, act: int) -> 'State':
        label = Action.to_label(act)
        return State(
                top = self.top,
                right = self.right,
                left = self.left.left,
                lchild1 = LabeledState(self.left, label),
                rchild1 = self.rchild1,
                lchild2 = self.lchild1,
                rchild2 = self.rchild2,
                sent = self.sent,
                prev = self,
                prevact = act)

    def reducer(self, act: int) -> 'State':
        label = Action.to_label(act)
        return State(
                top = self.left.top,
                right = self.right,
                left = self.left.left,
                lchild1 = self.left.lchild1,
                rchild1 = LabeledState(self, label),
                lchild2 = self.left.lchild2,
                rchild2 = self.left.rchild1,
                sent = self.sent,
                prev = self,
                prevact = act)

    def expand(self, act: int) -> 'State':
        act_type = Action.to_type(act)
        if act_type == Action.SHIFT:
            return self.shift(act)
        elif act_type == Action.REDUCEL:
            return self.reducel(act)
        elif act_type == Action.REDUCER:
            return self.reducer(act)
        else:
            raise RuntimeError(
                f'unexpected action id in State.expand: {act}')

    def to_heads(self) -> List[int]:
        def rec(s: State) -> None:
            if not s.prev.is_dummy:
                if not s.lchild1.state.is_dummy:
                    res[s.lchild1.state.top] = s.top
                if not s.rchild1.state.is_dummy:
                    res[s.rchild1.state.top] = s.top
                rec(s.prev)
        res = [-1] * len(self.sent)
        rec(self)
        return res

    def to_labels(self) -> List[int]:
        def rec(s: State) -> None:
            if not s.prev.is_dummy:
                act_type = Action.to_type(s.prevact)
                if act_type == Action.REDUCEL:
                    res[s.lchild1.state.top] = s.lchild1.label
                elif act_type == Action.REDUCER:
                    res[s.rchild1.state.top] = s.rchild1.label
                rec(s.prev)
        res = [0] * len(self.sent)
        rec(self)
        return res

    @property
    def is_dummy(self) -> bool:
        return self.top == 0 and self.right == 0

    @property
    def has_input(self) -> bool:
        return self.right < len(self.sent)

    @property
    def reducible(self) -> bool:
        return not self.left.is_dummy

    @property
    def is_final(self) -> bool:
        return not self.has_input and not self.reducible

    @staticmethod
    def of_sent(sent: Sentence) -> 'State':
        dummy_state = State._dummy(sent)
        dummy = LabeledState(dummy_state, UNK_ID)
        return State(
                top = 0,
                right = 1,
                left = dummy_state,
                lchild1 = dummy,
                rchild1 = dummy,
                lchild2 = dummy,
                rchild2 = dummy,
                sent = sent,
                prev = dummy_state,
                prevact = Action.NOOP)

    @staticmethod
    def _dummy(sent: Sentence) -> 'State':
        state = State(0, 0, None, None, # type: ignore
                None, None, None, sent, None, Action.NOOP)
        labeled = LabeledState(state, UNK_ID)
        state.left = state
        state.lchild1 = labeled
        state.rchild1 = labeled
        state.lchild2 = labeled
        state.rchild2 = labeled
        return state

    def __str__(self) -> str:
        def rec(s: State, stack: List[str]) -> List[str]:
            if not s.is_dummy:
                stack = rec(s.left, stack)
                stack.append(
                    f'{self.sent.words[self.top]}/{self.sent.tags1[self.top]}')
            return stack
        stack = rec(self, [])
        buff = [f'{word}/{tag1}' for word, tag1 in \
                    zip(self.sent.words[self.right:],
                        self.sent.tags1[self.right:]) ]
        return f'[{" ".join(stack)}] [{" ".join(buff)}]'

    def feature_set(self) -> FeatureSet:
        def idx_or_zero(idx: int) -> int:
            return idx if idx < len(self.sent) else 0

        idx = [idx_or_zero(self.right),                  # b0
              idx_or_zero(self.right + 1),               # b1
              idx_or_zero(self.right + 2),               # b2
              idx_or_zero(self.right + 3),               # b3
              self.top,                                  # s0
              self.lchild1.state.top,                    # s0l
              self.lchild2.state.top,                    # s0l2
              self.rchild1.state.top,                    # s0r
              self.rchild2.state.top,                    # s0r2
              self.lchild1.state.lchild1.state.top,      # s02l
              self.rchild1.state.rchild1.state.top,      # s12r
              self.left.top,                             # s1
              self.left.lchild1.state.top,               # s1l
              self.left.lchild2.state.top,               # s1l2
              self.left.rchild1.state.top,               # s1r
              self.left.rchild2.state.top,               # s1r2
              self.left.lchild1.state.lchild1.state.top, # s12l
              self.left.rchild1.state.rchild1.state.top, # s12r
              self.left.left.top,                        # s2
              self.left.left.left.top                    # s3
        ]

        labels = np.asarray(
                 [self.rchild1.label,                    # s0rc
                  self.rchild2.label,                    # s0rc2
                  self.lchild1.label,                    # s0lc
                  self.lchild2.label,                    # s0lc2
                  self.lchild1.state.lchild1.label,      # s02l
                  self.rchild1.state.rchild1.label,      # s02r
                  self.left.rchild1.label,               # s1rc
                  self.left.rchild2.label,               # s1rc2
                  self.left.lchild1.label,               # s1lc
                  self.left.lchild2.label,               # s1lc2
                  self.left.lchild1.state.lchild1.label, # s12l
                  self.left.rchild1.state.rchild1.label  # s12r
        ], int)

        words = self.sent.word_ids[idx]
        tags = self.sent.tag_ids[idx]
        return FeatureSet(words, tags, labels)


def oracle_states(
        sent: Sentence,
        action_size: int) -> List[TrainingExample]:
    def rec(s: State) -> None:
        if s.is_final: return
        if not s.reducible: 
            gold_action = Action.SHIFT
        else:
            s0h = sent.heads[s.top]
            s1h = sent.heads[s.left.top]
            if s1h == s.top:
                label = sent.label_ids[s.left.top]
                gold_action = Action.reducel(label)
            elif s0h == s.left.top and \
                all(h != s.top for h in sent.heads[s.right:]):
                label = sent.label_ids[s.top]
                gold_action = Action.reducer(label)
            else:
                gold_action = Action.SHIFT
        valid_actions = np.array( [s.is_valid(action) \
                for action in range(action_size)] ).astype(float)
        gold_action = np.array(gold_action, int)
        words, tags, labels = s.feature_set()
        res.append(
            (words, tags, labels, valid_actions, gold_action))
        rec(s.expand(gold_action))
    res: List[TrainingExample] = []
    state = State.of_sent(sent)
    rec(state)
    return res


########################################################################
############################  Neural Networks  #########################
########################################################################


class FeedForwardNetwork(chainer.Chain):
    def __init__(self,
            word_size: int,
            tag_size: int,
            label_size: int,
            action_size: int,
            embed_units: int = 50,
            hidden_units: int = 1024,
            rescale_embeddings: bool = True,
            pretrained: Optional[Array] = None) -> None:
        self.word_size = word_size
        self.tag_size = tag_size
        self.label_size = label_size
        self.action_size = action_size
        self.embed_units = embed_units
        token_context_size: int = 20
        label_context_size: int = 12
        self.contextsize = embed_units * \
                (token_context_size * 2 + label_context_size)
        super().__init__()
        with self.init_scope():
                self.w_embed = L.EmbedID(
                        self.word_size, embed_units, initialW=pretrained)
                self.t_embed = L.EmbedID(self.tag_size, embed_units)
                self.l_embed = L.EmbedID(self.label_size, embed_units)
                self.linear1 = L.Linear(self.contextsize, hidden_units)
                self.linear2 = L.Linear(hidden_units, self.action_size)

        # to ensure most ReLU units to activate in the first epochs
        self.linear1.b.data[:] = .2
        self.linear2.b.data[:] = .2

        if rescale_embeddings:
            self.rescale_embeddings(self.w_embed.W.data, .0, 1.)
            self.rescale_embeddings(self.t_embed.W.data, .0, 1.)
            self.rescale_embeddings(self.l_embed.W.data, .0, 1.)

    def rescale_embeddings(
            self, embed: Array, rmean: float, rstd: float) -> None:
        log('scaling embedding')
        mean = embed.mean()
        std  = embed.std()
        log(f'(mean={mean:0.4f}, std={std:0.4f}) ->')
        embed = (embed - mean) * rstd / std + rmean # type: ignore
        log(f'(mean={embed.mean():0.4f}, std={embed.std():0.4f})')

    def __call__(self,
            words: VariableOrArray,
            tags: VariableOrArray,
            labels: VariableOrArray,
            valid_actions: VariableOrArray,
            gold_actions) -> Variable:

        logits = self.forward(words, tags, labels, valid_actions)
        loss = F.softmax_cross_entropy(logits, gold_actions)
        accuracy = F.accuracy(logits, gold_actions, ignore_label=IGNORE)
        chainer.report({
            'loss': loss,
            'accuracy': accuracy
        }, self)
        return loss

    def forward(self,
            words: VariableOrArray,
            tags: VariableOrArray,
            labels: VariableOrArray,
            valid_actions: VariableOrArray) -> Variable:
        h = F.concat([
            self.w_embed(words),
            self.t_embed(tags),
            self.l_embed(labels),
        ], 1)
        h = F.relu(self.linear1(h))
        h = F.dropout(h, 0.5)
        h = self.linear2(h)
        return h * valid_actions


class ShiftReduceParser(object):
    def __init__(self, model: FeedForwardNetwork) -> None:
        self.model = model

    def _parse(self,
            sents: List[Sentence],
            batch_size: int = 1000) -> State:
        res = [None for _ in sents]
        queue = deque((i, State.of_sent(s)) for i, s in enumerate(sents))
        while len(queue) > 0:
            batch = [queue.pop() for _ in range(min(batch_size, len(queue)))]
            ids, states = zip(*batch)
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                preds = self.model.predict(states)
            for i, state, action in zip(ids, states, preds):
                state = state.expand(action)
                if state.isfinal:
                    res[i] = state
                else:
                    queue.append((i, state))
        assert (s is not None for s in res)
        return res

    def train(self,
            train_sents: List[Sentence],
            valid_sents: List[Sentence],
            action_size: int,
            out_dir: Path,
            init_model: Path,
            epoch: int = 10000,
            batch_size: int = 10000,
            gpu: int = -1,
            val_interval: Tuple[int, str] = (1000, 'iteration'),
            log_interval: Tuple[int, str] = (200, 'iteration'),
            ) -> None:

        if init_model:
            log('Load model from %s' % init_model)
            chainer.serializers.load_npz(init_model, self.model)

        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            self.model.to_gpu()

        TRAIN = [example for sent in train_sents \
                            for example in oracle_states(sent, action_size)]
        log(f'the size of training examples: {len(TRAIN)}')
        train_iter = chainer.iterators.SerialIterator(TRAIN, batch_size)

        VALID = [example for sent in valid_sents \
                            for example in oracle_states(sent, action_size)]
        log(f'the size of validation examples: {len(VALID)}')
        val_iter = chainer.iterators.SerialIterator(
                    VALID, batch_size, repeat=False, shuffle=False)

        optimizer = chainer.optimizers.AdaGrad(.01, 1e-6)
        optimizer.setup(self.model)
        optimizer.add_hook(WeightDecay(1e-8))

        updater = training.updaters.StandardUpdater(
            train_iter,
            optimizer,
            device = gpu,
            converter=chainer.dataset.concat_examples)

        trainer = training.Trainer(
            updater,
            (epoch, 'epoch'),
            out_dir)

        trainer.extend(
            extensions.Evaluator(
                val_iter,
                self.model,
                chainer.dataset.concat_examples,
                device = gpu),
            trigger = val_interval)

        trainer.extend(
            extensions.snapshot_object(
                self.model,
                'model_iter_{.updater.iteration}'),
            trigger=val_interval)

        trainer.extend(
                extensions.PrintReport([
            'epoch', 'iteration',
            'main/accuracy', 'main/loss',
            'validation/main/accuracy',
        ]), trigger=log_interval)

        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.run()

    def evaluate(self, sents: List[Sentence]) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(description='shift reduce parser')
    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers()

    train = subparsers.add_parser("train")
    train.add_argument('TRAIN', type = Path)
    train.add_argument('VALID', type = Path)
    train.add_argument('PATH', type = Path)
    train.add_argument('WORD', type = Path)
    train.add_argument('TAG', type = Path)
    train.add_argument('LABEL', type = Path)
    train.add_argument('--pretrained', type = Path)
    train.add_argument('--embed-units', type = int, default = 50)
    train.add_argument('--hidden-units', type = int, default = 1024)
    train.add_argument('--rescale-embeddings', type = bool, default = True)
    train.add_argument('--init-model', type = Path, default = None)
    train.add_argument('--epoch', type = int, default = 10000)
    train.add_argument('--batch-size', type = int, default = 10000)
    train.add_argument('--gpu', type = int, default = -1)
    args = parser.parse_args()

    loader = DataLoader(args.WORD, args.TAG, args.LABEL, args.pretrained)
    loader.save(args.PATH / 'loader.pickle')

    train_sents = loader.read_conll_train(args.TRAIN)
    for sent in train_sents:
        sent.projectivize()

    valid_sents = loader.read_conll_test(args.VALID)
    for sent in valid_sents:
        sent.projectivize()

    nn = FeedForwardNetwork(
            len(loader.words),
            len(loader.tags),
            len(loader.labels),
            loader.action_size,
            args.embed_units,
            args.hidden_units,
            args.rescale_embeddings,
            loader.pretrained
    )
    parser = ShiftReduceParser(nn)
    parser.train(
            train_sents,
            valid_sents,
            loader.action_size,
            args.PATH,
            args.init_model,
            args.epoch,
            args.batch_size,
            args.gpu,
            (2000, 'iteration'),
            (400, 'iteration'),
    )


if __name__ == '__main__':
    main()
