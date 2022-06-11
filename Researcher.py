import sys
import datetime

from Model import Model

sourcefiles = [
    'Database.py',
    'Embedder.py',
    'LSTM.py',
    'DISM.py',
    'fusion.py',
    'Model.py',
    'Parser.py',
    'global_transformer.py',
    'RBSM.py',
    'Researcher.py',
    'MultiFCN.py'
    'LayerNormalizer.py'
]


import os
from Database import Database

def GetTotalEffectiveLines(textFileList):
    database = Database('./Database/')
    sum = 0
    for filename in textFileList:
        path = os.path.join('./', filename)
        lines = database.GetListOfLines(path, addLastPeriod=False)
        cnt = 0
        for line in lines:
            if len(line.split()) > 0: cnt += 1
        sum += cnt

    return sum

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

logToFile = True

time = datetime.datetime.now()
print ('Start time: ', time)

if logToFile: sys.stdout = open('./Database/log.txt', 'w+')

start_time = datetime.datetime.now()
print ('Start time: ', start_time)
print("Total effective lines of source code: ", GetTotalEffectiveLines(sourcefiles))

metaParam = { 'embPath': './BERT_base_uncased/', 'dataPath': './Database/' }
metaParam ['emb_Layers'] = 12; metaParam['emb_dim'] = 300; metaParam['BiLSTM_dim'] = 300
O = Model( metaParams = metaParam, testMode = True)
O.Train(shuffleBuffer = 1000, miniBatchSize = 33, epochs = 3)
O.VisualizeTrainHistory()


end_time = datetime.datetime.now()
print ('End time: ', end_time)

#if logToFile: sys.stdout.close()