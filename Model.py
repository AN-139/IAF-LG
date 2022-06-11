import datetime
import tensorflow as tf
import numpy as np

from Database import Database
from Embedder import Embedder
from DISM import dism
from fusion import Fusion
from RBSM import rbsm
from global_transformer import Global_transformer
from LSTM import LSTM
from MultiFCN import MultiFCN
from VisualClass import VisualClass

class Model():
    def __init__(self, metaParams, testMode = False ):
        if metaParams == None or metaParams.get('dataPath', None) == None:
            print('Invalid Model Metaparameters (dataPath).')
            return
        self.database = Database(metaParams['dataPath'])
        dummyEmbedder = Embedder( bertPath = None, database = self.database )
        dummydism = dism(self.database, dummyEmbedder)
        dummyfusion = fusion(self.database, dummydism)
        dummyrbsm = rbsm(self.database, dummyEmbedder)
        dummyGlobal_transformer = Global_transformer(self.database, dummydism, dummyfusion, dummyrbsm)
	

    def __Create__(self, metaParams, testMode = False ):
        dataPath = metaParams['dataPath']
        embPath = metaParams['embPath']
        embLastNLayers = metaParams['emb_Layers']
        embFirstNHiddens = metaParams['emb_dim']
        LSTMDimHidden = metaParams['LSTM_dim']
        self.metaParms = metaParams
        self.database = Database(dataPath)
        self.embedder = Embedder(embPath, self.database, lastNLayers = embLastNLayers, firstNHiddens = embFirstNHiddens, testMode = testMode)
        self.metaParms['embedder'] = 'BERT'
        self.fusion = fusion(self.database, self.embedder, input = self.dism, testMode = testMode )
        self.rbsm = rbsm(self.database, self.embedder, dim_hidden = LSTMDimHidden, normalizeLayer = True, testMode = testMode)
        activations = ['tanh', 'tanh']
        self.multifcn = MultiFCN(dim_batch = 1, input = input, outputs = outputs, activations = activations, useLayerNormalizer = True)
        self.nWeightTensors = self.dism.nWeightTensors + self.fusion.nWeightTensors + self.rbsm.nWeightTensors + self.global_transformer.nWeightTensors + self.multifcn.nWeightTensors
        self.testMode = testMode
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002)
        self.weightsSnapshot = self.weights
        self.lossHistory = []
        self.scoreHistory = []

    def Initialize(self):
        self.database.Initialize()
        self.embedder.Initialize()
        self.dism.Initialize()
        self.fusion.Initialize()
        self.rbsm.Initialize()
        self.global_transformer.Initialize()
        self.multifcn.Initialize()
        self.lossHistory.clear()
        self.scoreHistory.clear()

    def Finalize(self):
        self.scoreHistory.clear()
        self.lossHistory.clear()
        self.multifcn.Finalize()
        self.global_transformer.Finalize()
        self.rbsm.Finalize()
        self.fusion.Finalize()
        self.dism.Finalize()
        self.embedder.Finalize()
        self.database.Finalize()

    def __GetWeightsTensorList(self):
        list = []
        list = list + self.dism.weights
        list = list + self.fusion.weights
        list = list + self.rbsm.weights
        list = list + self.global_transformer.weights
        list = list + self.multifcn.weights
        return list

    def __SetWeightsTensorList(self, list):
        cnt = 0
        self.dism.weights = list[cnt:]; cnt += self.dism.nWeightTensors
        self.fusion.weights = list[cnt:]; cnt += self.fusion.nWeightTensors
        self.rbsm.weights = list[cnt:]; cnt += self.rbsm.nWeightTensors
        self.global_transformer.weights = list[cnt:]; cnt += self.global_transformer.nWeightTensors
        self.multifcn.weights = list[cnt:]; cnt += self.multifcn.nWeightTensors 
        self.nWeightTensors == cnt
    weights = property(__GetWeightsTensorList, __SetWeightsTensorList)

    def NWeights(self):
        sum = 0
        for tensor in self.weights:
            n = 1; k = 0
            for _ in range(len(tensor.shape)):
                n = n * tensor.shape[k]
            sum += n   
        return sum

    def RemoveWeights(self):
        self.database.RemoveFile( self.database.pWeights )
    def SaveWeights(self):
        self.database.SaveBinaryData( self.weights, self.database.pWeights )
    def LoadWeights(self):
        if self.database.Exists(self.database.pWeights):
            self.weights = self.database.LoadBinaryData(self.database.pWeights)
    def Train(self, shuffleBuffer = 1000, miniBatchSize = 20, epochs = 5):
        if  self.metaParms.get('shuffle', None) != None and self.metaParms['shuffle'] != shuffleBuffer or \
            self.metaParms.get('MBSize', None) != None and self.metaParms['MBSize'] != miniBatchSize :
            self.RemoveCheckpointFiles()
        self.metaParms['shuffle'] = shuffleBuffer
        self.metaParms['MBSize'] = miniBatchSize
        self.metaParms['tensors'] = self.nWeightTensors
        self.metaParms['weights'] = self.NWeights()
        self.Initialize()
        dsTrain = self.database.CreateFullDataset(trainNotTest = True, addLastPeriod = True, returnSizeOnly = False)
        dsTrain = dsTrain.shuffle(buffer_size = shuffleBuffer, reshuffle_each_iteration = True).batch(batch_size = miniBatchSize, drop_remainder = True)
        accumEpoch = 0; step = 0
        self.weightsSnapshot = self.weights
        epochRange = range(epochs)

    def LogAccuracy(self, scoreHistory):
        hitRateList = self.database.GetHitRate(scoreHistory)
        precisionArray, recallArray = self.database.GetF1Score(scoreHistory)
        nClasses = len(self.database.tokenLabelStringList)
        print( 'Average hit rate : ', round( np.average(hitRateList), 2) )
        for c in range(nClasses):
            print( 'Average f1 precision for ', self.database.GetTokenLabelString(c), ' = ', round( np.nanmean(precisionArray[c, :]), 2) )
        for c in range(nClasses):
            print( 'Average f1 recall for ', self.database.GetTokenLabelString(c), ' = ', round( np.nanmean(recallArray[c, :]), 2) )

    def RemoveCheckpointFiles(self):
        self.RemoveWeights()
        self.database.RemoveFile(self.database.pTrainHistory)
    def OnStartEpoch(self, accumEpoch):
        self.LoadWeights()
        accumEpoch = 0
        if self.database.Exists(self.database.pTrainHistory):
            history = self.database.LoadBinaryData(self.database.pTrainHistory)
            self.metaParms, self.lossHistory, self.scoreHistory = history
            accumEpoch = self.metaParms['Epochs']
        return accumEpoch
    def OnEndEpoch(self, accumEpoch):
        history = (self.metaParms, self.lossHistory, self.scoreHistory)
        self.database.SaveBinaryData(history, self.database.pTrainHistory)
        self.SaveWeights()
    def LearnFromMiniBatch(self, batch, step):
        batchLoss = 0.0; scoreList = []
        sumGradient = []
        for weight in self.weightsSnapshot:
            sumGradient.append( tf.zeros_like( weight ) )
        batchSize = 0
        for dataset_record in batch:
            batchSize += 1
            with tf.GradientTape() as tape:
                tape.watch(self.weightsSnapshot)
                loss, score = self.GetLossForSingleExample(dataset_record, CacheTag.Train)
                batchLoss += loss.numpy()
                scoreList.append(score)
                tf.debugging.assert_all_finite(loss, message = 'Loss is a nan.')
                print( '\nloss =', loss.numpy() )
            grad = tape.gradient(loss, self.weightsSnapshot)
            for n in range(len(sumGradient)):
                if grad[n] is not None:
                    tf.debugging.assert_all_finite(grad[n], message = 'Gradient is a nan.')
                    sumGradient[n] = tf.add( sumGradient[n], grad[n] )
                else: pass # grad[n] is None. No change to sumGradient[n]
        self.optimizer.apply_gradients( zip(sumGradient, self.weightsSnapshot) )
        self.DoSomethingGreat()
        return sumGradient, batchLoss / batchSize, scoreList
    def DoSomethingGreat(self):
        pass
    def EvaluateOnTestData(self):
        batchLoss = 0.0; scoreList = []
        dsTest = self.database.CreateFullDataset(trainNotTest = False, addLastPeriod = True, returnSizeOnly = False)
        batchSize = 0
        for dataset_record in dsTest:
            batchSize += 1
            loss, score = self.GetLossForSingleExample(dataset_record, CacheTag.Test) # CacheTag.Test
            batchLoss += loss.numpy()
            scoreList.append(score)
            tf.debugging.assert_all_finite(loss, message = 'Loss is a nan.')
            print( '\nloss =', loss.numpy() )
        return batchLoss / batchSize, scoreList

    def GetLossForSingleExample(self, dataset_record, cacheTag):
        assert isinstance(cacheTag, CacheTag)
        sentence, aspect, sentiment, _, _, lineId = self.database.DecodeDatasetRecord(dataset_record)
        consistent, aspectList, sentimentList, wrongAspList, wrongOpnList = self.database.GetRefeinedLabels(sentence, aspect, sentiment, cacheTag, lineId)
        if self.testMode:
            if cacheTag == CacheTag.Train: tag = 'Train'
            else: tag = 'Test'
        scoreList = []
        lossTotal = tf.constant(value = 0.0, dtype = weightDType )
        for probDist, trueDist in zip(probDistList, trueDistList):
            a = - tf.multiply( trueDist, tf.math.log(probDist) )
            assert a.shape == [self.database.NLabelClass()]
            tf.debugging.assert_all_finite(a, message = 'a is a nan.')
            assert len(probDistList) > 0
            a = tf.reduce_sum(a)
            loss = a / ( 2.0 * len(probDistList) )
            tf.debugging.assert_all_finite(loss, message = 'Loss is a nan.')
            lossTotal = tf.add( lossTotal, loss )
            scoreList.append( (tf.argmax(trueDist).numpy(), tf.argmax(probDist).numpy() ) )
        return lossTotal, scoreList

    def Predict(self, sentence):
        probDistList = self.GetProbabilityDistributionList(sentence, CacheTag.Real, lineId = -1)
        predLabelStringList = [None] * len(probDistList)
        for tokenId in range(len(probDistList)):
            probDist = probDistList[tokenId]
            predClass = tf.argmax(probDist)
            assert predClass.shape == []
            predLabelString = self.database.GetTokenLabelString(predClass)
            predLabelStringList[tokenId] = predLabelString
        return predLabelStringList

    def GetProbabilityDistributionList(self, sentence, cacheTag, lineId):
        assert isinstance(cacheTag, CacheTag)
        
    def VisualizeTrainHistory(self):
        seriesDict = {}
        history = self.database.LoadBinaryData(self.database.pTrainHistory)
        metaParams, lossHistory, scoreHistory = history
        lossSereis = lossHistory
        seriesDict['loss'] = lossSereis
        hitRateSeries = self.database.GetHitRate(scoreHistory)
        seriesDict['hitRate'] = hitRateSeries
        precisionArray, recallArray = self.database.GetF1Score(scoreHistory)
        nClasses = len(self.database.tokenLabelStringList)
        for clsId in range(nClasses):
            seriesDict['p.' + self.database.tokenLabelStringList[clsId]] = precisionArray[clsId, :]
            seriesDict['r.' + self.database.tokenLabelStringList[clsId]] = recallArray[clsId, :]
        for clsId in range(nClasses) :
            assert len( precisionArray[clsId] ) == len(lossHistory)
        avgPrecision = [None] * len(lossHistory); avgRecall = [None] * len(lossHistory)
        for step in range(len(lossHistory)):
            avgPrecision[step] = np.nanmean( precisionArray[:, step] )
            avgRecall[step] = np.nanmean( recallArray[:, step] )
        seriesDict['avgPrecision'] = avgPrecision
        seriesDict['avgRecall'] = avgRecall
        vc = VisualClass()
        vc.PlotStepHistory(title, seriesDict, self.metaParms)

    def GetAverageHistory(self, lossHistory, scoreHistory):
        lossSereis = lossHistory
        avgLoss = 0.0; cnt = 0
        for loss in lossSereis:
            if loss is not None:
                avgLoss += loss
                cnt += 1
        assert cnt > 0
        avgLoss = avgLoss / cnt
        hitRateSeries = self.database.GetHitRate(scoreHistory)
        avgHitRate = 0.0; cnt = 0
        for hitRate in hitRateSeries:
            if hitRate is not None:
                avgHitRate += hitRate
                cnt += 1
        assert cnt > 0
        avgHitRate = avgHitRate / cnt
        return avgLoss, avgHitRate