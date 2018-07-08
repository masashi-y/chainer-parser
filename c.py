from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, Iterator
import asciitree
import chainer
from chainer import dataset, Variable
import chainer.functions as F
import chainer.links as L
from chainer import reporter
import numpy as np

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
    ('labels', VariableOrArray)
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
        assert entry not in res, \
            ('failure is Dataloader.read_vocab: '
            f'duplicate entry: {entry}.')
        res[entry] = len(res)
    return res


def read_pretrained(path: Path) -> Array:
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


class DataLoader(object):
    def __init__(self,
                word_path: Path,
                tag_path: Path,
                label_path: Path,
                pretrained_path: Optional[Path] = None) -> None:
        self.words = read_vocab(word_path)
        self.tags = read_vocab(tag_path)
        self.labels = read_vocab(label_path)
        self._accept_new_entries = True
        self.pretrained = read_pretrained(pretrained_path) \
                if pretrained_path is not None else None

    def _get_or_add_entry(self, entry: str) -> int:
        if entry in self.words:
            return self.words[entry]
        elif self._accept_new_entries:
            idx = len(self.words)
            self.words[entry] = idx
            return idx
        else:
            return UNK_ID

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


class Sentence(object):
    def __init__(self,
            words:  List[str],
            lemmas: List[str],
            tags1:  List[str],
            tags2:  List[str],
            labels: List[str],
            heads:  List[int],
            root_node = True) -> None:
        self.words  = words
        self.lemmas = lemmas
        self.tags1  = tags1
        self.tags2  = tags2
        self.labels = labels
        self.heads  = heads
        self.root_node = root_node
        self.orig_heads: Optional[List[int]] = None

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
            res += '\n'.join(items)
        return res

    def __len__(self) -> int:
        return len(self.words)

    @staticmethod
    def empty(root = True) -> 'Sentence':
        return Sentence(
            [PAD], [PAD], [UNK], [UNK], [UNK], [0], root_node = root)

    def append(self, word: str, lemma: str, tag1: str,
                     tag2: str, label: str, head: int) -> None:
        self.words.append(word)
        self.lemmas.append(lemma)
        self.tags1.append(tag1)
        self.tags2.append(tag2)
        self.labels.append(label)
        self.heads.append(head)

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

            if deepest_arc == -1: return
            lifted_head = self.heads[self.heads[deepest_arc]]
            self.heads[deepest_arc] = lifted_head


def read_conll(filename: Path) -> Iterator[Sentence]:
    """
    reads .conll file and returns a list of Sentence
    """
    sent = Sentence.empty(root = True)
    for line in filename.open():
        line = line.strip()
        if line == '':
            yield sent
            sent = Sentence.empty(root = True)
        items = line.split('\t')
        sent.append(
            word = items[1],
            lemma = items[2],
            tag1 = items[3],
            tag2 = items[4],
            label = items[7],
            head = int(items[6]),
        )


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
        empty = LabeledState(
            State._empty(self.sent),
            UNK_ID)
        return State(
                top = self.right,
                right = self.right + 1,
                left = self,
                lchild1 = empty,
                rchild1 = empty,
                lchild2 = empty,
                rchild2 = empty,
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
            if not s.prev.is_empty:
                if not s.lchild1.state.is_empty:
                    res[s.lchild1.state.top] = s.top
                if not s.rchild1.state.is_empty:
                    res[s.rchild1.state.top] = s.top
                rec(s.prev)
        res = [-1] * len(self.sent)
        rec(self)
        return res

    def to_labels(self) -> List[int]:
        def rec(s: State) -> None:
            if not s.prev.is_empty:
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
    def is_empty(self) -> bool:
        return self.top == 0 and self.right == 0

    @property
    def has_input(self) -> bool:
        return self.right < len(self.sent)

    @property
    def reducible(self) -> bool:
        return not self.left.is_empty

    @property
    def is_final(self) -> bool:
        return not self.has_input and not self.reducible

    @staticmethod
    def of_sent(sent: Sentence) -> 'State':
        empty_state = State._empty(sent)
        empty = LabeledState(empty_state, UNK_ID)
        return State(
                top = 0,
                right = 1,
                left = empty_state,
                lchild1 = empty,
                rchild1 = empty,
                lchild2 = empty,
                rchild2 = empty,
                sent = sent,
                prev = empty_state,
                prevact = Action.NOOP)

    @staticmethod
    def _empty(sent: Sentence) -> 'State':
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
            if not s.is_empty:
                stack = rec(s.left, stack)
                stack.append(
                    f'{self.sent.words[self.top]}/{self.sent.tags1[self.top]}')
            return stack
        stack = rec(self, [])
        buff = [f'{word}/{tag1}' for word, tag1 in \
                    zip(self.sent.words[self.right:],
                        self.sent.tags1[self.right:]) ]
        return f'[{" ".join(stack)}] [{" ".join(buff)}]'

    def sparse_features(self) -> FeatureSet:
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

        labels =  np.asarray(
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
        ], np.int32)

        words = self.sent.ws[idx]
        tags = self.sent.ts[idx]
        return FeatureSet(words, tags, labels)


def oracle_states(sent: Sentence) -> List[Tuple[FeatureSet, int]]:
    def rec(s: State) -> None:
        if s.is_final: return
        s0h = sent.heads[s.top]
        s1h = sent.heads[s.left.top]
        if not s.reducible: 
            gold_action = Action.SHIFT
        elif s1h == s.top:
            label = sent.ls[s.left.top]
            gold_action = Action.reducel(label)
        elif s0h == s.left.top and \
            all(h != s.top for h in sent.heads[s.right:]):
            label = sent.ls[s.top]
            gold_action = Action.reducer(label)
        else:
            gold_action = Action.SHIFT
        features = s.sparse_features()
        res.append((features, gold_action))
        rec(s.expand(gold_action))
    res: List[Tuple[FeatureSet, int]] = []
    state = State.of_sent(sent)
    rec(state)
    return res

########################################################################
############################  Neural Networks  #########################
########################################################################


class FeedForwardNetwork(chainer.Chain):
    def __init__(self,
            wordsize: int,
            tagsize: int,
            labelsize: int,
            targetsize: int,
            embed_units: int = 50,
            hidden_units: int = 1024,
            token_context_size: int = 20,
            label_context_size: int = 12,
            rescale_embeddings: bool = True,
            wscale: float = 1.,
            pretrained: Optional[Array] = None) -> None:
        self.wordsize = wordsize
        self.tagsize = tagsize
        self.labelsize = labelsize
        self.targetsize = targetsize
        self.embed_units = embed_units
        self.contextsize = embed_units * \
                (token_context_size * 2 + label_context_size)
        super().__init__()
        with self.init_scope():
                self.w_embed = L.EmbedID(
                        self.wordsize, embed_units, pretrained=pretrained)
                self.t_embed = L.EmbedID(self.tagsize, embed_units)
                self.l_embed = L.EmbedID(self.labelsize, embed_units)
                self.linear1 = L.Linear(
                        self.contextsize, hidden_units, wscale = wscale)
                self.linear2 = L.Linear(
                        hidden_units, self.targetsize, wscale = wscale)

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
        h  = F.concat([
            self.w_embed(words),
            self.t_embed(tags),
            self.l_embed(labels),
        ], 1)
        h = F.relu(self.linear1(h))
        h = F.dropout(h, ratio=0.5)
        h = self.linear2(h)
        return h * valid_actions

