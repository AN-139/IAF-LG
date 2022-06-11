import os
import json
import pickle
import tensorflow as tf

class Database():

    sentencesFname = 'sentences'
    aspectLabelsFname = 'aspect_labels'
    sentimentLabelsFname = 'sentiment_labels'
    annotationSuffix = '_ann'
    cleanSuffix = '_clean'
    removedSuffix = '_removed'
    trainPrefix = 'train_'
    testPrefix = 'test_'
    parameters = 'parameters'
    weights = 'weights'
    combined = 'combined'
    tokenLableClass = 'tokenLabelClass'
    history = 'history'
    cache = 'cache'
    databaseSuffix = '_database'
    embedderSuffix = '_embedder'
    log = 'log'
    dataFolder = ''

    def __init__(self, homePath):
        self.homePath = homePath

        absHomePath = os.path.abspath(self.homePath)
        assert os.path.exists(absHomePath)

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname) + '.txt'
        assert os.path.exists(path)
        self.pTrainSentences = path
        
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname) + '.txt'
        assert os.path.exists(path)
        self.pTestSentences = path

        self.pTrainAnnotations = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname + self.annotationSuffix) + '.txt'
        #assert os.path.exists(self.pathToTrainAnnotations)
        self.pTestAnnotations = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname + self.annotationSuffix) + '.txt'
        #assert os.path.exists(self.pathToTestAnnotations)
        
        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.aspectLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTrainAspectLables = path
        
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.aspectLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTestAspectLables = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentimentLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTrainOpinionLables = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentimentLabelsFname) + '.txt'
        assert os.path.exists(path)
        self.pTestOpinionLables = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname + self.cleanSuffix) + '.txt'
        self.pTrainSentencesClean = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname + self.cleanSuffix) + '.txt'
        self.pTestSentencesClean = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.sentencesFname + self.removedSuffix) + '.txt'
        self.pTrainSentencesRemoved = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.sentencesFname + self.removedSuffix) + '.txt'
        self.pTestSentencesRemoved = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.combined) + '.txt'
        self.pTrainCombined = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.combined) + '.txt'
        self.pTestCombined = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.tokenLableClass) + '.txt'
        self.pTrainTokenLabelClass = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.tokenLableClass) + '.txt'
        self.pTestTokenLabelClass = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.history) + '.bin'
        self.pTrainHistory = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.history) + '.bin'
        self.pTestHistory = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.cache + self.databaseSuffix) + '.bin'
        self.pTrainCacheDatabase = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.cache + self.databaseSuffix) + '.bin'
        self.pTestCacheDatabase = path

        path = os.path.join( absHomePath, self.dataFolder, self.trainPrefix + self.cache + self.embedderSuffix) + '.bin'
        self.pTrainCacheEmbedder = path
        path = os.path.join( absHomePath, self.dataFolder, self.testPrefix + self.cache + self.embedderSuffix) + '.bin'
        self.pTestCacheEmbedder = path


        path = os.path.join( absHomePath, self.parameters) + '.txt'
        self.pParameters = path

        path = os.path.join( absHomePath, self.weights) + '.bin'
        self.pWeights = path

        path = os.path.join( absHomePath, self.log) + '.txt'
        self.pLog = path
        
        self.tokenLabelStringList = \
            [self.possenlb, self.negsenlb, self.neusenlb]

        self.trainSize = self.CreateFullDataset(trainNotTest = True, addLastPeriod = True, returnSizeOnly = True)
        self.testSize = self.CreateFullDataset(trainNotTest = False, addLastPeriod = True, returnSizeOnly = True)

        self.trainCache = [None] * self.trainSize
        self.testCache = [None] * self.testSize

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def OnStartEpoch(self, epoch):
        self.LoadCache()

    def OnEndEpoch(self, epoch):
        self.SaveCache()

    def RemoveFile(self, path):
        try:
            os.remove(path)
        except:
            print("Error while deleting file ", path)

    def RemoveCache(self):
        self.RemoveFile(self.pTrainCacheDatabase)
        self.RemoveFile(self.pTestCacheDatabase)

    def LoadCache(self):
        if self.Exists(self.pTrainCacheDatabase):
            self.trainCache = self.LoadBinaryData(self.pTrainCacheDatabase)
        if self.Exists(self.pTestCacheDatabase):
            self.testCache = self.LoadBinaryData(self.pTestCacheDatabase)

    def SaveCache(self):
        self.SaveBinaryData(self.trainCache, self.pTrainCacheDatabase)
        self.SaveBinaryData(self.testCache, self.pTestCacheDatabase)

    def GetListOfLines(self, path, addLastPeriod = False):
        try:
            file = open(path, 'rt')
            text = file.read()
            file.close()
        except:
            raise Exception("Couldn't open/read/close file: " + path)
        
        lines = text.split('\n')

        nlines = []
        if addLastPeriod == True:
            for line in lines:
                if line[-1] != '?' and line[-1] != '.' :
                    line = line + '.'
                nlines.append(line)

        if addLastPeriod == False:
            return lines
        else:
            return nlines

    def SaveJsonData(self, data, path):
        try:
            file = open(path, 'wt+')
            json.dump(data, file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)

    def LoadJsonData(self, path):
        try:
            file = open(path, 'rt')
            data = json.load(file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)
        
        return data

    def SaveBinaryData(self, data, path):
        try:
            file = open(path, 'wb+')
            pickle.dump(data, file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)

    def LoadBinaryData(self, path):
        try:
            file = open(path, 'rb')
            data = pickle.load(file)
            file.close()
        except:
            raise Exception("Couldn't open/read/close file:" + path)

        return data

    def Exists(self, path):
        return os.path.exists(path)


    def GetHitRate(self, scoreHistory):
        hitRate = []
        for batchScore in scoreHistory:
            cntHit = 0; cntMiss = 0
            for sentenceScore in batchScore:
                for trueClass, predClass in sentenceScore:
                    if trueClass != 0:  # excluding the defalut label.
                        if trueClass == predClass: cntHit += 1
                        else: cntMiss += 1
            hitRate.append( 1.0 * cntHit / (cntHit + cntMiss + 1) )

        return hitRate

    def GetF1Score(self, scoreHistory) :
        classes = len(self.tokenLabelStringList)
        batches = len(scoreHistory)
        precision = [ [None] * batches ] * classes; recall = [ [None] * batches ] * classes

        for batchId in range(batches) :
            batchScore = scoreHistory[batchId]

            relavant = [None] * classes; truePositive = [None] * classes; falsePositive = [None] * classes

            for sentenceScore in batchScore :
                for trueClass, predClass in sentenceScore:
                    if trueClass != 0:  # excluding the defalut label.
                        if relavant[trueClass] == None : relavant[trueClass] = 1
                        else: relavant[trueClass] += 1
                        if trueClass == predClass :
                            if truePositive[trueClass] == None : truePositive[trueClass] = 1
                            else: truePositive[trueClass] += 1
                        else:
                            if falsePositive[predClass] == None : falsePositive[predClass] = 1 # Note predClass.
                            else: falsePositive[predClass] += 1 # Note predClass
        
            for clsId in range(classes):
                if clsId != 0:
                    if truePositive[clsId] != None :
                        if falsePositive[clsId] != None :
                            precision[clsId][batchId] = 1.0 * truePositive[clsId] / (truePositive[clsId] + falsePositive[clsId])
                        if relavant[clsId] != None :
                            recall[clsId][batchId] = 1.0 * truePositive[clsId] / relavant[clsId]

        return precision, recall