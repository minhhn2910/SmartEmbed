import os
import pickle
import numpy as np
from gensim.models.fasttext import FastText
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import distance
import sys
import os.path
sys.path.append('./contract_level/Normalize/')
from contract_normalization import Contract_Norm
sys.path.append('./contract_level/Vectorize/')
from contract_vectorize import Contract_Vec
sys.path.append('./contract_level/Crawl')
from contract_crawl import Contract_Detail
sys.path.append('./todo/')
from config import Config
from clone_detect import Clone_Detect

class SmartEmbed(object):

    def __init__(self):

        self.USERINPUT = "./todo/USERINPUT/"
        self.contract_model_path = "./contract_level/Model/FastText/fasttext_model"
        self.fasttext_model = self.get_fasttext_model()

    def get_fasttext_model(self):
        '''
            load fasttext model
        '''
        FASTTEXT_MODEL = FastText.load(self.contract_model_path)
        return FASTTEXT_MODEL

    def save_to_file(self, user_in):
        '''
            save user input into file
        '''
        if not os.path.exists( self.USERINPUT ):
            os.makedirs(self.USERINPUT)
        with open( self.USERINPUT + 'user_input.sol', 'w') as handle:
            handle.write(user_in)

    def parser(self):
        '''
            parse the user_input.sol into AST
        '''
        cmd = "java -classpath ./contract_level/Parse/antlr4.jar:./contract_level/Parse/target/ \
                    Tokenize ./todo/USERINPUT/user_input.sol ./todo/CONTRACT_RESULT/"
        os.system(cmd)

    def normalizer(self):
        '''
            normalize
        '''
        # print("entering normalizer...")
        cn = Contract_Norm("./todo/CONTRACT_RESULT/")
        return cn.line_span, cn.normalized_tokens

    def vectorizer(self, norm_result):
        # print("entering vectorizer...")
        cv = Contract_Vec(norm_result, self.fasttext_model)
        return cv.vector

    def get_vector(self, user_input):
        self.save_to_file(user_input)
        self.parser()
        norm_result = self.normalizer()
        vec_result = self.vectorizer(norm_result)
        return vec_result

    def get_similarity(self, vec1, vec2):
        numerator = distance.cdist(vec1.reshape(1, 150), vec2, 'euclidean')
        denominator = np.linalg.norm(vec1, axis=1) + np.linalg.norm(vec2)
        similarity = 1 - np.divide(numerator, denominator)
        return similarity[0][0]

def get_similarity( vec1, vec2):
    numerator = distance.cdist(vec1.reshape(1, 150), vec2.reshape(1, 150), 'euclidean')
    denominator = np.linalg.norm(vec1) + np.linalg.norm(vec2)
    similarity = 1 - np.divide(numerator, denominator)
    return similarity[0][0]
def vectorize(model, normed_tokens, dimension=150):
    embedding_vector = np.empty((0,dimension), dtype="float64")
    # norm_tokens : 1D array [number of tokens]
    failed_tokens = 0
    statement_vector = np.zeros((dimension,), dtype="float64")
    for token in normed_tokens:
        if token in model:
            statement_vector = np.add(statement_vector, model[token])
        else:
            failed_tokens += 1
            continue
    print ("failed tokens:", failed_tokens)
    tmp_embedding_vector = np.vstack([statement_vector])
    embedding_vector = np.append(embedding_vector,tmp_embedding_vector,axis=0)
    return embedding_vector

def simple_tokenizer(text):
    return text.split(" ")
def main():
    assert (len(sys.argv) == 3), "Usage: python get_embedding.py <sentence1> <sentence2>"
    sentence1 = sys.argv[1]
    sentence2 = sys.argv[2]
    fasttext_model = FastText.load("./statement_level/Model/FastText/fasttext_model")
    normed_tokens1 = simple_tokenizer(sentence1)
    normed_tokens2 = simple_tokenizer(sentence2)
    print ("normed_tokens1:", normed_tokens1)
    print ("normed_tokens2:", normed_tokens2)
    vectorized_tokens1 = vectorize(fasttext_model, normed_tokens1)
    vectorized_tokens2 = vectorize(fasttext_model, normed_tokens2)
    similarity = get_similarity(vectorized_tokens1, vectorized_tokens2)
    #similarity = se.get_similarity(contract_vector1, contract_vector2)
    print("similarity score:", similarity)



if __name__ == '__main__':
    main()
