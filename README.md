# A demo project to understand how Seq2Seq works with character sequences #

code source: 
https://github.com/udacity/deep-learning/tree/master/seq2seq

description:
input a character sequence with length <=7, output the reversed sequence.
E.g. input = 'apple', output='aelpp'

modifications to source code:
1. the source and target dictionary are not necessarily generated from training samples, using alpabet to generate a common dictionary will improve efficiency during training
2. no attention mechanism is original code, apply attention mechanism to model

Environment:
Tensorflow 1.3
