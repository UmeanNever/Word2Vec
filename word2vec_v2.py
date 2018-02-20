import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit
from io import open
import operator
import matplotlib.pyplot as plt


#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.
#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.


#... (4) Re-train the algorithm using different context windows.


#... (5) Compare cosine similarities between learned word vectors.










#.................................................................................
#... global variables
#.................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from






#.................................................................................
#... load in the data and convert tokens to one-hot indices
#.................................................................................



def loadData(filename):
    global uniqueWords, wordcodes, wordcounts
    override = False
    if override:
        #... for debugging purposes, reloading input file and tokenizing is quite slow
        #...  >> simply reload the completed objects. Instantaneous.
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
        uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        return fullrec


    #... load in the unlabeled data file. You can load in a subset for debugging purposes.
    handle = open(filename, "r", encoding="utf8")
    fullconts =handle.read().split("\n")
    fullconts = [entry.split("\t")[1].replace("<br />", "") for entry in fullconts[1:(len(fullconts)-1)]]

    #... apply simple tokenization (whitespace and lowercase)
    fullconts = [" ".join(fullconts).lower()]





    print ("Generating token stream...")
    #... populate fullrec as one-dimension array of all tokens in the order they appear.
    #... ignore stopwords in this process
    #... for simplicity, you may use nltk.word_tokenize() to split fullconts.
    #... keep track of the frequency counts of tokens in origcounts.
    fullrec = nltk.word_tokenize(fullconts[0])
    stop_words = set(stopwords.words('english'))
    fullrec = [w for w in fullrec if not w in stop_words]
    min_count = 50
    origcounts = Counter(fullrec)





    print ("Performing minimum thresholding..")
    #... populate array fullrec_filtered to include terms as-is that appeared at least min_count times
    #... replace other terms with <UNK> token.
    #... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)
    fullrec_filtered = [w if origcounts[w] > min_count else '<UNK>' for w in fullrec ]
    wordcounts = Counter(fullrec_filtered)

    #... after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec = fullrec_filtered






    print ("Producing one-hot indicies")
    #... (TASK) sort the unique tokens into array uniqueWords
    #... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
    #... replace all word tokens in fullrec with their corresponding one-hot indices.
    uniqueWords = list(zip(*sorted(wordcounts.items(), key=operator.itemgetter(1), reverse=True))[0])
    wordcodes = {key: value for (key, value) in zip(uniqueWords, xrange(0, len(uniqueWords)))}
    fullrec = [wordcodes[x] for x in fullrec]





    #... close input file handle
    handle.close()



    #... store these objects for later.
    #... for debugging, don't keep re-tokenizing same data in same way.
    #... just reload the already-processed input data with pickles.

    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))


    #... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
    return fullrec







#.................................................................................
#... compute sigmoid value
#.................................................................................

@jit(nopython=True)
def sigmoid(x):
    return float(1)/(1+np.exp(-x))









#.................................................................................
#... generate a table of cumulative distribution of words
#.................................................................................


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
    #global wordcounts
    #... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    max_exp_count = 0


    print ("Generating exponentiated count vectors")
    #... for each uniqueWord, compute the frequency of that word to the power of exp_power
    #... store results in exp_count_array.
    exp_count_array = [wordcounts[x]**exp_power for x in uniqueWords]
    max_exp_count = sum(exp_count_array)



    print ("Generating distribution")

    #... compute the normalized probabilities of each term.
    #... using exp_count_array, normalize each value by the total value max_exp_count so that
    #... they all add up to 1. Store this corresponding array in prob_dist
    prob_dist = np.array(exp_count_array, dtype='float') / max_exp_count





    print ("Filling up sampling table")
    #... create a dict of size table_size where each key is a sequential number and its value is a one-hot index
    #... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
    #... multiplied by table_size.
    cumulative_dict = {}
    table_size = 1e8
    index = 0
    cumulative_prob = prob_dist[0]*table_size
    for i in range(1, int(table_size) + 1):
        if i > int(cumulative_prob) and len(prob_dist) > index+1:
            index += 1
            cumulative_prob += prob_dist[index]*table_size
        cumulative_dict[i] = index




    return cumulative_dict






#.................................................................................
#... generate a specific number of negative samples
#.................................................................................


def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter
    results = []
    #... randomly sample num_samples token indices from samplingTable.
    #... don't allow the chosen token to be context_idx.
    #... append the chosen indices to results
    i = 0
    while i<num_samples:
        id = random.randint(1, len(samplingTable))
        if id == context_idx:
            continue
        else:
            i += 1
            results.append(samplingTable[id])


    return results







@jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token, sequence_chars,W1,W2,negative_indices):
    # sequence chars was generated from the mapped sequence in the core code
    nll_new = 0.
    summation = np.zeros(hidden_size, dtype=np.float64)
    for k in range(0, len(sequence_chars)):
        #... implement gradient descent. Find the current context token from sequence_chars
        #... and the associated negative samples from negative_indices. Run gradient descent on both
        #... weight matrices W1 and W2.
        #... compute the total negative log-likelihood and store this in nll_new.
        codes = sequence_chars[k]
        summation += (sigmoid(np.dot(W2[codes], W1[center_token])) - 1) * W2[codes]
        W2[codes] = W2[codes] - learning_rate*(sigmoid(np.dot(W2[codes], W1[center_token])) - 1)*W1[center_token]

        for i in range(0, num_samples):
            j = negative_indices[2*k + i]
            summation += (sigmoid(np.dot(W2[j], W1[center_token])) - 0) * W2[j]
            W2[j] = W2[j] - learning_rate * (sigmoid(np.dot(W2[j], W1[center_token])) - 0) * W1[center_token]

    W1[center_token] = W1[center_token] - learning_rate*summation

    for k in range(0, len(sequence_chars)):
        codes = sequence_chars[k]
        nll_new += -np.log(sigmoid(np.dot(W2[codes], W1[center_token])))
        for i in range(0, num_samples):
            j = negative_indices[2*k + i]
            nll_new += -np.log(sigmoid(np.dot(-W2[j], W1[center_token])))

    return [nll_new]






#.................................................................................
#... learn the weights for the input-hidden and hidden-output matrices
#.................................................................................


def trainer(curW1 = None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
    vocab_size = len(uniqueWords)           #... unique characters
    hidden_size = 100                       #... number of hidden neurons
    #context_window = [1, 2, 3, 4]
    #context_window = [-4, -3, -2, -1]
    context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations


    #... determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window)))
    end_point = len(fullsequence)-(max(max(context_window),0))
    mapped_sequence = fullsequence



    #... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    if curW1==None:
        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
    else:
        #... initialized from pre-loaded file
        W1 = curW1
        W2 = curW2



    #... set the training parameters
    epochs = 5
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0




    #... Begin actual training
    for j in range(0,epochs):
        print ("Epoch: ", j)
        prevmark = 0

        #... For each epoch, redo the whole sequence...
        for i in range(start_point,end_point):

            if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                print ("Progress: ", round(prevmark+0.1,1))
                prevmark += 0.1
            if iternum%10000==0 and iternum>0:
                print ("Negative likelihood: ", nll)
                nll_results.append(nll)
                nll = 0


            #... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
            center_token = mapped_sequence[i]

            #... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
            if center_token == '<UNK>':
                continue




            iternum += 1
            #... now propagate to each of the context outputs
            mapped_context = [mapped_sequence[i+ctx] for ctx in context_window]
            negative_indices = []
            for q in mapped_context:
                negative_indices += generateSamples(q, num_samples)
            [nll_new] = performDescent(num_samples, learning_rate, center_token, mapped_context, W1,W2, negative_indices)
            nll += nll_new




    for nll_res in nll_results:
        print (nll_res)
    plt.plot(np.arange(1, len(nll_results)+1), nll_results)
    plt.xlabel('iteration number (x 1e4)')
    plt.ylabel('neg loglikelihood')
    plt.show()
    return [W1,W2]



#.................................................................................
#... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
#.................................................................................

def load_model():
    handle = open("saved_W1(p6_1).data","rb")
    W1 = np.load(handle)
    handle.close()
    handle = open("saved_W2(p6_1).data","rb")
    W2 = np.load(handle)
    handle.close()
    return [W1,W2]






#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................

def save_model(W1,W2):
    handle = open("saved_W1.data","wb+")
    np.save(handle, W1, allow_pickle=False)
    handle.close()

    handle = open("saved_W2.data","wb+")
    np.save(handle, W2, allow_pickle=False)
    handle.close()






#... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
#... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
#... vector predict similarity to a context word.






#.................................................................................
#... code to start up the training function.
#.................................................................................
word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
    else:
        curW1 = None
        curW2 = None
    [word_embeddings, proj_embeddings] = trainer(curW1,curW2)
    save_model(word_embeddings, proj_embeddings)









#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................

def morphology(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [word_seq[0], # suffix averaged
    embeddings[wordcodes[word_seq[1]]]]
    vector_math = vectors[0]+vectors[1]
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.







#.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................

def analogy(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [embeddings[wordcodes[word_seq[0]]],
    embeddings[wordcodes[word_seq[1]]],
    embeddings[wordcodes[word_seq[2]]]]
    vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
    #... find whichever vector is closest to vector_math
    #... (TASK) Use the same approach you used in function prediction() to construct a list
    #... of top 10 most similar words to vector_math. Return this list.







#.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................


def prediction(target_word):
    global word_embeddings, uniqueWords, wordcodes
    targets = [target_word]
    outputs = []
    similarities = []
    target_code = wordcodes[target_word]
    for word in uniqueWords:
        word_code = wordcodes[word]
        similarities.append(1 - cosine(word_embeddings[word_code],word_embeddings[target_code]))

    top_index = sorted(range(len(similarities)), key=lambda i: similarities[i])[-11:-1]
    for index in top_index:
        outputs.append({'word': uniqueWords[index], 'score': similarities[index]})
    #... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
    #... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
    #... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
    #... return a list of top 10 most similar words in the form of dicts,
    #... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}

    return outputs











if __name__ == '__main__':
    if len(sys.argv)==2:
        filename = sys.argv[1]
        #filename = 'unlabeled-text.tsv'
        #... load in the file, tokenize it and assign each token an index.
        #... the full sequence of characters is encoded in terms of their one-hot positions

        fullsequence= loadData(filename)
        print ("Full sequence loaded...")
        #print(uniqueWords)
        #print (len(uniqueWords))



        #... now generate the negative sampling table
        print ("Total unique words: ", len(uniqueWords))
        print("Preparing negative sampling table")
        samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)


        #... we've got the word indices and the sampling table. Begin the training.
        #... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
        #... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
        #... ... and uncomment the load_model() line

        #train_vectors(preload=False)
        [word_embeddings, proj_embeddings] = load_model()








        #... we've got the trained weight matrices. Now we can do some predictions
        targets = [u"good", u"bad", u"scary", u"funny"]
        target_result = []
        for targ in targets:
            print("Target: ", targ)
            bestpreds= (prediction(targ))
            for pred in bestpreds:
                print (pred["word"],":",pred["score"])
                target_result.append({'target_word': targ, 'similar_word': pred['word'], 'similar_score': pred['score']})
            print ("\n")

        with open('output.txt', 'wb') as csvfile:
            fieldname = ['target_word', 'similar_word', 'similar_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldname, lineterminator='\n')
            writer.writeheader()
            writer.writerows(target_result)


        #... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
        print (analogy(["son", "daughter", "man"]))
        print (analogy(["thousand", "thousands", "hundred"]))
        print (analogy(["amusing", "fun", "scary"]))
        print (analogy(["terrible", "bad", "amazing"]))



        #... try morphological task. Input is averages of vector combinations that use some morphological change.
        #... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
        #... the morphology() function.

        s_suffix = [word_embeddings[wordcodes["stars"]] - word_embeddings[wordcodes["star"]]]
        others = [["types", "type"],
        ["ships", "ship"],
        ["values", "value"],
        ["walls", "wall"],
        ["spoilers", "spoiler"]]
        for rec in others:
            s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
        s_suffix = np.mean(s_suffix, axis=0)
        print (morphology([s_suffix, "techniques"]))
        print (morphology([s_suffix, "sons"]))
        print (morphology([s_suffix, "secrets"]))


        similar_result = []
        with open('intrinsic-test_v2.tsv') as f1:
            reader = csv.reader(f1, delimiter='\t', quoting=csv.QUOTE_NONE)
            next(reader)
            for row in reader:
                word1 = wordcodes[row[1]]
                word2 = wordcodes[row[2]]
                similarity = cosine(word_embeddings[word1], word_embeddings[word2])
                similar_result.append({'id': row[0], 'similarity': similarity})

        with open('intrinsic-test_res.csv', 'wb') as csvfile:
            fieldname = ['id', 'similarity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldname, lineterminator='\n')
            writer.writeheader()
            writer.writerows(similar_result)



    else:
        print ("Please provide a valid input filename")
        sys.exit()


