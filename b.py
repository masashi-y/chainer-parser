
from a import  *
from itertools import chain

class StructuredParser(Parser):
    def __init__(self, vocab, beamsize=32, batchsize=30,
            niters=2000, evaliter=20, **args):
        super(StructuredParser, self).__init__(vocab, **args)
        self.beamsize  = beamsize
        self.batchsize = batchsize
        self.niters    = niters
        self.evaliter  = evaliter

    def expand_with_score(self, s, act, score):
        res = s.expand(act)
        res.score = score
        return res

    def beamsearch(self, batch):
        """
        batch: list of System instances
        """
        charts = map(lambda s: [[s]], batch)
        alive = [-1] * len(batch)

        k = 0
        while any(v == k-1 for v in alive):
            offset = 0
            batch = list(chain.from_iterable(
                [chart[k] for chart in charts]))
            exs = map(lambda s: Example.gen_test(
                s, self.targetsize), batch)
            preds = self.model(exs)
            for cid in range(len(charts)):
                if cid > 0: offset += len(charts[cid-1][k])
                chart = charts[cid]
                states = chart[k]
                if states[0].isfinal:
                    chart.append(states)
                else:
                    alive[cid] = k
                    nexts = []
                    for sid in range(len(states)):
                        bid = offset + sid
                        valid = exs[bid].valid
                        expnd = [self.expand_with_score(states[sid],
                                act, preds[bid, act:act+1])
                                for act in range(self.targetsize)
                                if valid[act] > .0]
                        nexts.extend(expnd)
                    nexts.sort(key=lambda s: s.score.data[0], reverse=True)
                    chart.append(nexts[:self.beamsize])
            k += 1
        return charts, alive
