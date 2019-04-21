# File: config.py
# Author: Ronil Pancholia
# Date: 4/20/19
# Time: 6:37 PM

## dropouts
encInputDropout = 0.2 # dropout of the rnn inputs to the Question Input Unit
encStateDropout = 0.0 # dropout of the rnn states of the Question Input Unit
stemDropout = 0.2 # dropout of the Image Input Unit (the stem)
qDropout = 0.08 # dropout on the question vector
qDropoutOut = 0 # dropout on the question vector the goes to the output unit
memoryDropout = 0.15 # dropout on the recurrent memory
readDropout = 0.15 # dropout of the read unit
writeDropout = 1.0 # dropout of the write unit
outputDropout = 0.85 # dropout of the output unit
controlPreDropout = 1.0 # dropout of the write unit
controlPostDropout = 1.0 # dropout of the write unit
wordEmbDropout = 1.0 # dropout of the write unit