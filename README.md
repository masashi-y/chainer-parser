# chainer-parser

a.py:
a reimplementation of the following paper:
A Fast and Accurate Dependency Parser using Neural Networks, Chen and Manning,  EMNLP 2014

using relu nonlinearity (instead of cubic one) and AdaGrad(alpha=0.01, epsilon=1e-6) achives
UAS:0.9155	LAS:0.8941
in 60000 iterations.

a.py also implements one with the same model and different training strategy,
which is one found in syntaxnet graph\_builder.
This uses momentum SGD, exponential decaying of learning rate, and Weight averaging.
those techniques are explained in the following paper.
Structured Training for Neural Network Transition-Based Parsing, Weiss et al., ACL 2015
In this setting, the model fails to improve after reaching 90.0 UAS point.

b.py tries to implement globally normalized one, but not done yet.
Globally Normalized Transition-Based Neural Networks, Andor et al., ACL 2016
