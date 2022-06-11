
import torch
import numpy as np # for mean() only.
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch, BertTokenizer, BertModel
#from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

import os

class Embedder():

    useLargeBert = False
   
    def __init__(self, bertPath, database, lastNLayers = 3, firstNHiddens = 50, testMode = False):
        self.database = database

        if bertPath == None:
            self.dummyMode = True
            return
        else:
            self.dummyMode = False

        """
        lastNLayers <= 12 for bert base, <= 24 for large bert
        firstNHiddens <= 768 for beas bert, <= 1024 for large bert
        """
        assert 0 <= lastNLayers and lastNLayers <= 12
        assert 0 <= firstNHiddens and firstNHiddens <= 768

        self.lastNLayers = lastNLayers
        self.firstNHiddens = firstNHiddens
        self.dim_wordVector = lastNLayers * firstNHiddens

        assert os.path.isdir(bertPath)

        abs_bert_path = os.path.abspath(bertPath)
        abs_ckptPath = os.path.join(abs_bert_path, 'bert_model.ckpt')
        abs_configPath = os.path.join(abs_bert_path, 'bert_config.json')
        abs_modelPath = os.path.join(abs_bert_path, 'pytorch_model.bin')

        # Do not assert.
        #assert os.path.exists(abs_ckptPath)
        #assert os.path.exists(abs_configPath)
        #assert os.path.exists(abs_modelPath)

        self.Tokenizer = BertTokenizer.from_pretrained(abs_bert_path, cache_dir=None)

        convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
            abs_ckptPath, abs_configPath, abs_modelPath )

        if self.useLargeBert == False:
            self.Model = BertModel.from_pretrained(abs_bert_path)
        else:
            self.Model = BertModel.from_pretrained('bert-large-uncased')

        self.trainCache = [{} for _ in range(self.database.trainSize)] 
        self.testCache = [{} for _ in range(self.database.testSize)]

        self.testMode = testMode

        self.Model.eval() 

    def Initialize(self):
        pass

    def Finalize(self):
        pass

    def OnStartEpoch(self, epoch):
        self.LoadCache()

    def OnEndEpoch(self, epoch):
        self.SaveCache()

    def RemoveCache(self):
        self.database.RemoveFile(self.database.pTrainCacheEmbedder)
        self.database.RemoveFile(self.database.pTestCacheEmbedder)

    def LoadCache(self):
        if self.database.Exists(self.database.pTrainCacheEmbedder):
            self.trainCache = self.database.LoadBinaryData(self.database.pTrainCacheEmbedder)
        if self.database.Exists(self.database.pTestCacheEmbedder):
            self.testCache = self.database.LoadBinaryData(self.database.pTestCacheEmbedder)

    def SaveCache(self):
        self.database.SaveBinaryData(self.trainCache, self.database.pTrainCacheEmbedder)
        self.database.SaveBinaryData(self.testCache, self.database.pTestCacheEmbedder)
        
    def Tokenize(self, text):
        # tokenize as expected by the model
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.Tokenizer.tokenize(marked_text)
        indexed_tokens = self.Tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.LongTensor([indexed_tokens])
        segments_tensors = torch.LongTensor([segments_ids])
        
        return tokens_tensor, segments_tensors

    def GetTokenList(self, text, cacheTag, lineId, lineFeed = False):
        """
        Return: excludes "[CLS]" / "[SEP]"
        """
        cache = None
        if cacheTag == CacheTag.Train: cache = self.trainCache
        elif cacheTag == CacheTag.Test: cache = self.testCache

        if cache is not None and cache[lineId].get('TL', None) != None:
            tokenList, tString = cache[lineId]['TL']
        else:
            nText = text
            if nText[-1] == '\n':
                nText = nText[:-1]
            tokenList = self.Tokenizer.tokenize(text)

            tString = ''
            for token in tokenList:
                tString += (' ' + token + ',')
            tString = tString[1:-1]
            if lineFeed:
                tString += '\n'

            if cache is not None and cache[lineId].get('TL', None) == None:
                cache[lineId]['TL'] = (tokenList, tString)

        return tokenList, tString

    def LookupForTokenId(self, token, tokensListSen):
        tokenId = -1
        for id in range(len(tokensListSen)):
            if tokensListSen[id] == token:
                tokenId = id; break
        if tokenId < 0:
            raise Exception('Token not found.')
        else:
            return tokenId

    def GetEmbLayersExtended(self, text):
        """Return: ndarrys of shape (layers, tokens, hiddens)"""

        tokens_tensor, segments_tensors = self.Tokenize(text)

        # Predict hidden states features for each layer
        with torch.no_grad():
            layers, _ = self.Model(tokens_tensor, segments_tensors)
        
        layers = np.array([torch.Tensor.numpy(layer) for layer in layers]) # layers[layer, sentence, token, hidden]
        layers = layers[:, 0, :, :] # Squeeze automatically the sentence dimension. We have a single sentence.
        layers = layers[-self.lastNLayers:, :, :] # resulting in [layer, token, hidden]

        return layers

    def GetWordVector(self, embLayersExtended, tokenId):
        assert 0 <= tokenId and tokenId < embLayersExtended.shape[1] - 1 # [layer, token, hidden].
        assert embLayersExtended.shape[1] >= 3 # Read the line below.
        embLayers = embLayersExtended[:, :, :self.firstNHiddens]

        tid = tokenId + 1
        #if tid > 1 : tid -= 1
        embLayers = embLayers[:, tid, :] # token dimension is squeezed out automatically, resulting in [layer, hidden]

        dim = embLayers.shape[0] * embLayers.shape[1]
        assert dim == self.dim_wordVector
        vector = embLayers.reshape(dim,)

        return vector

    def AverageWordVector(self, embLayersExtended, tokenIdList):
        assert len(tokenIdList) > 0
        sumVector = np.zeros( shape = (self.dim_wordVector), dtype = np.float) # np.ndarray(shape = (self.dim_wordVector), dtype=np.float) # defaults to 0.0
        for tokenId in tokenIdList:
            vector = self.GetWordVector(embLayersExtended, tokenId)
            assert not np.isnan( vector ).any()
            sumVector += vector
            assert sumVector.shape == (self.dim_wordVector,)
        
        return sumVector / len(tokenIdList)

    def GenerateEmbedding(self, sentence):
        embedding = self.GetEmbLayersExtended(sentence)

        return embedding

    def MapToSubstrings(self, token, candSubstrings):
        mapping = []; token = token.lower()

        concat = ''
        for candId in range(len(candSubstrings)):
            substring = candSubstrings[candId]

            if len(substring) >= 2 and substring[0] == '#' and substring[1] == '#':
                substring = substring[2:]

            concat += substring

            if token == concat :
                for id in range(candId + 1) :
                    mapping.append(id)
                break

        return mapping

    def GetTokenMapping(self, parTokens, embTokens, cacheTag, lineId):
        cache = None
        if cacheTag == CacheTag.Train: cache = self.trainCache
        elif cacheTag == CacheTag.Test: cache = self.testCache

        if cache is not None and cache[lineId].get('TM', None) != None:
            compatible, mapping = cache[lineId]['TM']
        else:
            compatible = False; mapping = []

            hop = 1
            while compatible == False and  hop < len(embTokens):
                compatible, mapping = self.GetTokenMapingWithHop(parTokens, embTokens, hop)
                hop += 1

            if cache is not None and cache[lineId].get('TM', None) == None:
                cache[lineId]['TM'] = (compatible, mapping)
        
        if not compatible :
            compatible = compatible # for breakpoing

        return compatible, mapping

    def  GetTokenMapingWithHop(self, parTokens, embTokens, hop = 1):
        compatible = True 
        mapping = [None] * len(parTokens) 
        searchPoint = 0    

        for parId in range(len(parTokens)):

            nestMap = self.MapToSubstrings(parTokens[parId], embTokens[searchPoint:])
            if len(nestMap) > 0 :
                for id in nestMap: nestMap[id] += searchPoint
            elif parId + 1 < len(parTokens) and searchPoint + hop < len(embTokens):
                for convPoint in range(searchPoint + hop, len(embTokens)):
                    convMap = self.MapToSubstrings(parTokens[parId + 1], embTokens[convPoint:])
                    if len(convMap) > 0:
                        for id in range(len(convMap)): convMap[id] += convPoint
                        for id in range(searchPoint, convMap[0]):
                            nestMap.append(id)
                        break # stop convolution.
            if len(nestMap) > 0 :
                mapping[parId] = nestMap
                searchPoint = nestMap[-1] + 1
            else:
                compatible = False
                break
        
        if searchPoint < len(embTokens):
            compatible = False
        
        return compatible, mapping

    def Norm(self, array):
        return np.sum(array * array) ** .5
    def Related(self, array_a, array_b):
        return np.sum(array_a * array_b) / max(self.Norm(array_a), self.Norm(array_b)) ** 2

    def MeanOverTokens(self, layers):
        """  encoded_layers: ndarray of shape (#layers, token, hiddensize) """
        return np.mean(layers, axis = 1)
        
    def Compare(self, seq_a, seq_a1, seq_b, seq_b1 = -1):
        vector_a = self.GetEmbLayersExtended(seq_a); vector_a = self.MeanOverTokens(vector_a)
        vector_a1 = self.GetEmbLayersExtended(seq_a1); vector_a1 = self.MeanOverTokens(vector_a1)
        
        related_aa1 = self.Related(vector_a, vector_a1)

        print('type of rep:', type(vector_a), vector_a.shape)
        
        vector_b = self.GetEmbLayersExtended(seq_b); vector_b = self.MeanOverTokens(vector_b)
        if seq_b1 == -1: 
            seq_b1 = seq_a1; vector_b1 = vector_a1
        else: 
            vector_b1 = self.GetEmbLayersExtended(seq_b1); vector_b1 = self.MeanOverTokens(vector_b1)
        
        related_bb1 = self.Related(vector_b, vector_b1)
        
        if related_aa1 > related_bb1: # don't use absolute.
            text = "'" + seq_a + "' : '" + seq_a1 + "' > \n'" + seq_b + "' : '" + seq_b1 + "'."
            rate = related_aa1 / related_bb1
            high = related_aa1; low = related_bb1
        else:
            text = "'" + seq_b + "' : '" + seq_b1 + "' > \n'" + seq_a + "' : '" + seq_a1 + "'." 
            high = related_bb1; low = related_aa1
            rate = high / low
            
        print(text)
        print( 'Rate = %0.3f / %0.3f = %0.2f' % (high, low, abs(rate)) )
        
        return text, high, low, rate


"""
# Unit test.
O = Embedder('./BERT_base_uncased/', lastNLayers = 2, firstNHiddens = 5)
text = "I can't go."
tokens, segments = O.Tokenize(text)
print(type(tokens), type(segments))

text = "I don't go."
tList, tString = O.GetTokenList(text, False)

layersExt = O.GetEmbLayersExtended(text)
print(type(layersExt), 'layers: ', len(layersExt))
for layer in layersExt:
    print(type(layersExt), layersExt.shape) # (hidden,)

vector = O.GetWordVector(layersExt, tokenId = 3)
print(type(vector), 'layers: ', vector.shape)
print(vector)

text, high, low, rate = O.Compare('I like him.', 'He is gentle.', 'I hate him.')
text, high, low, rate = O.Compare('I like him.', 'The earch is round.', 'I hate him.')

"""