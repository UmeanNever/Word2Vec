
# Word2vec Model with Neural Network

**by Yuming Yang**

In this project, I implement a basic version of word2vec that will learn word representations from context information and then try using those representations in some application scenarios like intrinsic tasks that measure word similarity.

## Trainning and Testing Data
I train this model on the **unlabeled-text.tsv** file and test mainly on **intrinsic-test_v2.tsv**.

## Codes
I integrate all the codes in **word2vec.py**. To start the word2vec training or see some small tests, try

```
python word2vec.py unlabeled-text.tsv
```

If you want to train on new text files or parameters, you can comment the default saved varibale setting in the code and assign new values. 

## Solution Detail
+ First load in the data source and tokenize into one-hot vectors.
Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.

+ Prepare a negative sampling distribution table to draw negative samples from.
Consistent with the original word2vec paper, this distribution should be exponentiated.

+ Run a training function for a number of epochs to learn the weights of the hidden layer.
This training will occur through backpropagation from the context words down to the source word.

![nll](https://github.com/UmeanNever/Word2Vec/blob/master/p5.png "negative likelihood plot")

+ Re-train the algorithm using different context windows. 

+ Compare cosine similarities between learned word vectors.

## Reference

https://arxiv.org/abs/1402.3722
https://arxiv.org/pdf/1411.2738.pdf