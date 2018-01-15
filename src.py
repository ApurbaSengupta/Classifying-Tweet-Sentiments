'''
Date: 29 October 2017
@authors: Apurba Sengupta, Dhruv Desai

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, re
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse, interpolate

start_time1 = time.time()

# **********************************************************************************************
#
#   function definitions
#    
# **********************************************************************************************
    

def loadTrainData(filename):

    '''
    Question 1.1

    For the first part you are required to download the Sentiment140 dataset found
    here (http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip). Unzip the 
    file and read in the CSV with training data (it has 1.6 million entries). The   
    file contains a table of Sentiment, UserID, Date, no-query, user id, the actual
    tweet. Read in only the sentiment and the tweet. You are welcome to use other 
    values as additional features. Sentiment in the data file is classified as
    either 0 (negative emotion) or 4 (positive emotion). Convert these to -1 and 1
    respectively.

    '''

    print "\n Loading the training data ... "

    # read the data into a pandas DataFrsme object and convert it into a NumPy matrix
    train_data_matrix = np.array(pd.read_csv(filename, header = None))

    # the last column of the matrix is the tweet, taken only upto the first comma in the tweet
    X_train = np.array([tweet.split(',')[0] for tweet in train_data_matrix[:,-1]])

    # the first column of the matrix is the sentiment (either 0 or 4)
    Y_train = train_data_matrix[:,0]

    # label the sentimnents with value 4 as 1
    Y_train[Y_train==4] = 1

    # label the sentiments with value 0 as -1
    Y_train[Y_train==0] = -1

    end_time1 = time.time() - start_time1
    
    print "\n Training data Loaded ..."

    print "\n Time taken to load the training data (in seconds) = ", end_time1

    start_time2 = time.time()

    '''
    Question 1.2 : cleaning up the data 
    
    Perform the following operations on each of the tweets.
    1. Convert all letters into lowercase.
    2. Convert all occurences of 'www.' or 'https://' or 'http://' to 'URL'.
    3. Remove additional white spaces.
    4. Remove all punctuation.
    5. Replace @username with AT-USER.
    6. Replace duplicate words - e.g. 'very very' should be replaced by 'very'.
    7. After Steps 1-6 have been performed, check if the tweet has any of the words
    from stopwords.txt (provided on Canvas). If there are any other words that are
    stop words, ignore them. Whatever now remains of the original tweet is the 
    feature vector for that tweet. Create a table of feature vectors and sentiment.
    For example for the first tweet in the dataset, which looks like "@switchfoot
    http://twitpic.com/2y1zl - Awww, that's a bummer. You shoulda got David Carr 
    of Third Day to do it. ;D" the extracted feature is "aww". The second tweet 
    "is upset that he can't update his Facebook by texting it... and might cry as a
    result School today also. Blah!" after doing steps 1-5 yields the features
    ['upset', 'update', 'facebook', 'texting', 'cry', 'result', 'school']
    
    '''

    print "\n\n Cleaning the training data ... \n\n"

    # convert the tweets into String objects
    X_train = X_train.astype(str)

    print "\n Converting the tweets to lowercase ..."

    # 1. convert all the tweets into lowercase
    X_train = np.core.defchararray.lower(X_train)
    
    print "\n Converting all occurences of 'www.', 'https://' and 'http://' to 'URL' ..."

    # 2. convert all occurences of 'www.','https://','http://' to 'URL'
    X_train = np.array([re.sub(r'((http://)|(www.)|(https://))(?:[a-zA-Z]|[0-9]|[$-_~@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', tweet) for tweet in X_train])
    
    print "\n Removing additional white spaces at beginning of text and in between text ..."

    # 3. remove additional white spaces at beginning of text and in between text
    X_train = np.core.defchararray.strip(X_train)
    X_train = np.core.defchararray.replace(X_train, '  ', ' ')

    print "\n Removing all punctuation ..."

    # 4. remove all punctuation
    X_train = np.array([re.sub(r'[!"#$%&\'\(\)*+,-./:;<=>?\[\\\]^_`\{|\}~]', '', tweet) for tweet in X_train])

    print "\n Converting all occurances of '@<username>' to 'AT-USER' ..."

    # 5. convert all occurances of @username to 'AT-USER'
    X_train = np.array([re.sub(r'(@)(?:[a-zA-Z]|[0-9])+', 'AT-USER', tweet) for tweet in X_train])             
    
    print "\n Replacing consecutive duplicate words with a single occurance of the word ..."

    # 6. replace consecutive duplicate words with a single occurance
    X_train = np.array([re.sub(r'\b(\w+)(\s\1\b)+', r'\1', tweet) for tweet in X_train])

    print "\n Removing stop words ..."

    # 7. remove stop words
    with open(r"stopwords.txt", 'r') as stopWordsFile:
        wordList = stopWordsFile.read().split('\n')
        regex = re.compile(r'\b(%s)\b' % '|'.join(wordList))
        X_train = np.array([re.sub(regex, ' ', tweet) for tweet in X_train])

    print "\n\n Cleaned the training data ...\n\n"
    
    '''
    Question 1.3

    Extract unigram features from the bag of words. The bag of words here is the 
    set of all words collected after performing steps 1-6. You are free to use any
    library to create unigram features from the bag of words. To give an example 
    (from wikipedia) : Consider we have the tweets "John likes to watch movies. 
    Mary likes movies too." and the tweet "John also likes to watch football 
    games". The bag of words (Steps 1-6) gives us (John,likes,to,watch,movies,Mary,
    too,also,football,games.) To now extract unigram features, in the tweet "John 
    likes to watch movies. Mary likes movies too." count the number of times each 
    word appears in the bag of words. Thus, the feature vector would look like 
    [1, 2, 1, 1, 2, 1, 1, 0, 0, 0]. John occurs once in the tweet, likes occurs 
    twice and so on. For more detail see the wikipedia article on bag of words. 
    You are free to extract bi-gram/n-gram features. Thus, at the end of this you 
    should have a list of features and the sentiment attached to these features.

    '''

    print "\n Creating bags-of-words (BoW) of the features ..."

    # generate bags-of-words (BoW) of the features
    X_train = np.core.defchararray.split(X_train)
    train_bags_of_words = np.array([list(set(tweetWords)) for tweetWords in X_train]) 

    print "\n BoW of the features created ..."

    print "\n\n Creating table of BoW of features and corresponding sentiments ...\n\n"

    # generate table of BoW of features and corresponding sentiments
    data_table = pd.DataFrame(train_bags_of_words, columns = ['Features'])
    data_table['Sentiments'] = Y_train
    print "\n", data_table

    end_time2 = time.time() - start_time2

    print "\n\n Table of BoW of features and corresponding sentiments created ..."

    print "\n\n Time taken to clean the training data to form table of BoW of features and corresponding sentiments (in seconds) = ", end_time2

    return X_train, Y_train

def loadTestData(filename):

    start_time6 = time.time()

    '''
    Question 1.6

    Read in the test CSV dataset. Perform all steps in Question 1.1 and 1.2. An
    additional step required is that the sentiments in the test set are 0, 2 and 4.
    Convert the tweets with sentiment 2 to 4. Report test accuracy and a plot of 
    test error vs number of iterations for classfiers trained in Part 1.4 and Part 1.5.
    
    '''

    print "\n\n Loading the test data ... "

    # read the data into a pandas DataFrsme object and convert it into a NumPy matrix
    test_data_matrix = np.array(pd.read_csv(filename, header = None))

    # the last column of the matrix is the tweet, taken only upto the first comma in the tweet
    X_test = np.array([tweet.split(',')[0] for tweet in test_data_matrix[:,-1]])

    # the first column of the matrix is the sentiment (either 0 or 4)
    Y_test = test_data_matrix[:,0]

    # label the sentimnents with value 2 as 4
    Y_test[Y_test==2] = 4
    
    # label the sentimnents with value 4 as 1
    Y_test[Y_test==4] = 1

    # label the sentiments with value 0 as -1
    Y_test[Y_test==0] = -1

    end_time6 = time.time() - start_time6
    
    print "\n Test data Loaded ..."

    print "\n\n Time taken to load the test data (in seconds) = ", end_time6

    start_time7 = time.time()
    
    print "\n\n Cleaning the test data ... \n\n"
    
    # convert the tweets into String objects
    X_test = X_test.astype(str)
    
    print "\n\n Converting the tweets to lowercase ..."

    # 1. convert all the tweets into lowercase
    X_test = np.core.defchararray.lower(X_test)

    print "\n Converting all occurences of 'www.', 'https://' and 'http://' to 'URL' ..."

    # 2. convert all occurences of 'www.','https://','http://' to 'URL'
    X_test = np.array([re.sub(r'((http://)|(www.)|(https://))(?:[a-zA-Z]|[0-9]|[$-_~@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', tweet) for tweet in X_test])

    print "\n Removing additional white spaces at beginning of text and in between text ..."

    # 3. remove additional white spaces at beginning of text and in between text
    X_test = np.core.defchararray.strip(X_test)
    X_test = np.core.defchararray.replace(X_test, '  ', ' ')
    
    print "\n Removing all punctuation ..."

    # 4. remove all punctuation
    X_test = np.array([re.sub(r'[!"#$%&\'\(\)*+,-./:;<=>?\[\\\]^_`\{|\}~]', '', tweet) for tweet in X_test])

    print "\n Converting all occurances of '@<username>' to 'AT-USER' ..."

    # 5. convert all occurances of @username to 'AT-USER'
    X_test = np.array([re.sub(r'(@)(?:[a-zA-Z]|[0-9])+', 'AT-USER', tweet) for tweet in X_test])             
    
    print "\n Replacing consecutive duplicate words with a single occurance of the word ..."

    # 6. replace consecutive duplicate words with a single occurance
    X_test = np.array([re.sub(r'\b(\w+)(\s\1\b)+', r'\1', tweet) for tweet in X_test])

    print "\n Removing stop words ..."

    # 7. remove stop words
    with open(r"stopwords.txt", 'r') as stopWordsFile:
        wordList = stopWordsFile.read().split('\n')
        regex = re.compile(r'\b(%s)\b' % '|'.join(wordList))
        X_test = np.array([re.sub(regex, ' ', tweet) for tweet in X_test])

    print "\n\n Cleaned the test data ...\n\n"

    # get words of each tweet
    X_test = np.core.defchararray.split(X_test)
    
    end_time7 = time.time() - start_time7

    print "\n\n Time taken to clean the test data (in seconds) = ", end_time7
    
    return X_test, Y_test

def createUnigramFeatureMatrixFromTweets(X_train, X_test):

    start_time3 = time.time()
    
    '''
    Question 1.3

    Extract unigram features from the bag of words. The bag of words here is the 
    set of all words collected after performing steps 1-6. You are free to use any
    library to create unigram features from the bag of words. To give an example 
    (from wikipedia) : Consider we have the tweets "John likes to watch movies. 
    Mary likes movies too." and the tweet "John also likes to watch football 
    games". The bag of words (Steps 1-6) gives us (John,likes,to,watch,movies,Mary,
    too,also,football,games.) To now extract unigram features, in the tweet "John 
    likes to watch movies. Mary likes movies too." count the number of times each 
    word appears in the bag of words. Thus, the feature vector would look like 
    [1, 2, 1, 1, 2, 1, 1, 0, 0, 0]. John occurs once in the tweet, likes occurs 
    twice and so on. For more detail see the wikipedia article on bag of words. 
    You are free to extract bi-gram/n-gram features. Thus, at the end of this you 
    should have a list of features and the sentiment attached to these features.

    Question 1.5

    Use AdaGrad to train a classifier on the features extracted above. Report a plot
    of training error vs. number of iterations for every 1000 iterations. Merge this
    plot with the one from the previous question. No libraries are allowed here.

    '''

    # encode the string entries as 'latin-1' and get back the preprocessed tweets
    train_tweets_latin = np.array([[word.decode('latin-1') for word in tweet] for tweet in X_train])
    preprocessed_tweet_corpus_train = [' '.join(tweet) for tweet in train_tweets_latin]

    # encode the string entries as 'latin-1' and get back the preprocessed tweets
    test_tweets_latin = np.array([[word.decode('latin-1') for word in tweet] for tweet in X_test])
    preprocessed_tweet_corpus_test = [' '.join(tweet) for tweet in test_tweets_latin]
    
    # corpus of preprocessed tweets
    preprocessed_tweet_corpus = preprocessed_tweet_corpus_train + preprocessed_tweet_corpus_test

    print "\n\n Creating unigram features and converting the data into a matrix of feature occurances ..."

    # convert the preprocessed tweets to a matrix of unigram feature counts
    vectorizer = CountVectorizer(ngram_range = (1,1))
    
    # training set matrix
    X_train_sparse = vectorizer.fit_transform(preprocessed_tweet_corpus)[:X_train.shape[0],:]
    
    # test set amtrix
    X_test_sparse = vectorizer.fit_transform(preprocessed_tweet_corpus)[X_train.shape[0]:,:]

    print "\n\n Converted the data into training and test matrices of feature occurances ..."

    end_time3 = time.time() - start_time3

    print "\n\n Time taken to construct the training and test set matrices (in seconds) = ", end_time3

    return X_train_sparse, X_test_sparse

def runPegasosAndAdaGrad(X_train_sparse, Y_train):

    start_time4 = time.time()

    '''
    Question 1.4

    Use PEGASOS to train an SVM on the features extracted above. Make a plot of
    training error v/s number of iterations.

    '''

    # **************************************************************************************************
    #
    #                       PEGASOS - Primal Estimated sub-GrAdient SOlver for SVM 
    #
    #   Input : Training Set S = {(x1, y1), (x2, y2), ....., (xn, yn)}, Regularization parameter lambda,
    #           Number of iterations = T  
    #
    #   Initialize : w such that norm(w) <= 1/sqrt(lambda)
    #
    #   For t = 1, 2, ..., T:
    #
    #       1. Choose A (a randomly chosen subset of S) with |A| = B
    #
    #       2. A+ = {(x,y) (present in A) such that y<w,x> < 1}
    #
    #       3. learning rate = eta = 1/(count * lambda)   
    #
    #       4. gradient = lambda * w - (eta / B) * <y,x> for (x,y) in A+
    #
    #       5. new w = w - eta * gradient
    #
    #       6. projection of new w = new w * min(1, 1/(sqrt(lambda) * norm(new w))
    #
    # **************************************************************************************************

    print "\n\n\n\n Running PEGASOS on the training data ...\n"

    # set number of iterations
    T_P = 5000

    # set lambda (regularizer)
    regularizer_P = 0.00001

    # set number of random rows to be selected at every iteration
    B_P = 1000

    # initialize count
    count_P = 1

    # initialize decision boundary
    w_P = np.zeros(X_train_sparse.shape[1])

    # initial prediction
    Y_pred_P = X_train_sparse.dot(w_P)

    # initial training error
    training_error_P = float(np.logical_xor((Y_pred_P > 0).astype(int), (Y_train.astype(int) > 0).astype(int)).sum())/float(X_train_sparse.shape[0])

    # lsit to hold training error2
    training_error_list_P = [training_error_P]

    # list to hold number of iterations
    count_list_P = [count_P]
    
    # list to hold the decision values for each iteration
    w_P_list = [w_P]
    
    for i in range(T_P):
    
        # randomly generate B_P row numbers
        j = np.random.randint(0, X_train_sparse.shape[0] , B_P)
        
        # generate the corresponding random subsets of X_train_sparse and Y_train 
        x_P = X_train_sparse[j]
        y_P = Y_train[j].astype(int)
    
        # compute the learning rate
        eta_P = 1/float(count_P * regularizer_P)
    
        # see the sign of y<w,x> and generate a binary decision vector after checking if y<w,x> < 1 
        decision_vector_P = np.multiply(x_P.dot(w_P), y_P)  
        decision_vector_P = (decision_vector_P < 1).astype(int)
    
        # convert decision vector to a sparse CSR vector
        decision_vector_sparse_P = sparse.csr_matrix(decision_vector_P).T
        
        # filter training examples for which y<w,x> >= 1
        x_new_P = x_P.multiply(decision_vector_sparse_P)
        y_new_P = np.multiply(y_P, decision_vector_P)
    
        # compute the gradient
        grad_P = np.subtract(np.multiply(w_P, regularizer_P), np.multiply(x_new_P.T.dot(y_new_P), float(eta_P)/float(B_P)))
    
        # compute new decision boundary
        w_new_P = np.subtract(w_P, np.multiply(grad_P, eta_P))
    
        # take projection of decision boundary so that it does not exit the set 1/srqt(regularizer_P)
        w_new_proj_P = np.multiply(w_new_P, min(1, 1/(float(np.sqrt(regularizer_P)) *  float(np.linalg.norm(w_new_P)))))
    
        # increment the counter and put it in count_list
        count_P = count_P + 1
        count_list_P.append(count_P)
        
        # decision boundary is new decision boundary, put it in w_P_list
        w_P = w_new_proj_P
        w_P_list.append(w_P)
        
        # calculate training error and put it in training_error_list
        y_pred_P = x_P.dot(w_P)
        
        training_error_P = float(np.logical_xor((y_pred_P > 0).astype(int), (y_P > 0).astype(int)).sum())/float(B_P)
        training_error_list_P.append(training_error_P)
    
    print "\n\n PEGASOS run successfully on the training data ...\n"

    end_time4 = time.time() - start_time4

    print "\n\n Time taken to train data using PEGASOS (in seconds) = ", end_time4
    
    print "\n\n\n Plotting Training Error v/s Number of Iterations for the trained model for PEGASOS ...\n"

    # plot training error v/s number of iterations
    plt.title('\n Training Error v/s Number of Iterations\n')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Training Error')
    plt.plot(count_list_P, training_error_list_P, color = '#e0ffff')
    tck1 = interpolate.splrep(count_list_P, training_error_list_P, k = 3, s = 900)
    training_error_list_P_int = interpolate.splev(count_list_P, tck1, der = 0)
    plt.plot(count_list_P, training_error_list_P_int, color = 'blue', label = 'PEGASOS')
    plt.legend()
    plt.show()

    start_time5 = time.time()

    '''
    Question 1.5

    Use AdaGrad to train a classifier on the features extracted above. Report a plot
    of training error vs. number of iterations for every 1000 iterations. Merge this
    plot with the one from the previous question. No libraries are allowed here.

    '''

    # **************************************************************************************************
    #
    #                                   AdaGrad - Adaptive Gradient 
    #
    #   Input : Training Set S = {(x1, y1), (x2, y2), ....., (xn, yn)}, Regularization parameter lambda,
    #           Number of iterations = T 
    #
    #   Initialize : w such that norm(w) <= 1/sqrt(lambda), S old = (1,1,1.....,1)
    #
    #   For t = 1, 2, ..., T:
    #
    #       1. Choose A (a randomly chosen subset of S)
    #
    #       2. A+ = {(x,y) (present in A) such that y<w,x> < 1}
    #
    #       3. learning rate = eta = 0.01   
    #
    #       4. gradient = - <y,x> for (x,y) in A+
    #
    #       5. S new = S old + gradient^2
    #
    #       6. new w = w - [eta / (B * sqrt(S new))] * gradient
    #   
    #       7. projection of new w = new w * min(1, 1/(sqrt(lambda) * norm(S new * new w))
    #
    # **************************************************************************************************

    print "\n\n\n\n Running AdaGrad on the training data ...\n"

    # set number of iterations
    T_A = 5000

    # set lambda (regularizer)
    regularizer_A = 0.005

    # set number of random rows to be selected at every iteration
    B_A = 1000
    
    # set the learning rate
    eta_A = 0.01
    
    # initialize count
    count_A = 1

    # initialize decision boundary
    w_A = np.zeros(X_train_sparse.shape[1])

    # initialize vector of diagonal elements of tranformation matrix G
    S_A = np.ones(X_train_sparse.shape[1])

    # initial prediction
    Y_pred_A = X_train_sparse.dot(w_A)

    # initial training error
    training_error_A = float(np.logical_xor((Y_pred_A > 0).astype(int), (Y_train.astype(int) > 0).astype(int)).sum())/float(X_train_sparse.shape[0])

    # lsit to hold training error2
    training_error_list_A = [training_error_A]

    # list to hold number of iterations
    count_list_A = [count_A]
    
    # list to hold the decision values for each iteration
    w_A_list = [w_A]

    for i in range(T_A):
        
        # randomly generate B_A row numbers
        j = np.random.randint(0, X_train_sparse.shape[0], B_A)
    
        # generate the corresponding random subsets of X_train_sparse and Y_train 
        x_A = X_train_sparse[j]
        y_A = Y_train[j].astype(int)
        
        # see the sign of y<w,x> and generate a binary decision vector after checking if y<w,x> < 1 
        decision_vector_A = np.multiply(x_A.dot(w_A), y_A)  
        decision_vector_A = (decision_vector_A < 1).astype(int)
        
        # convert decision vector to a sparse CSR vector
        decision_vector_sparse_A = sparse.csr_matrix(decision_vector_A).T
        
        # filter training examples for which y<w,x> >= 1
        x_new_A = x_A.multiply(decision_vector_sparse_A)
        y_new_A = np.multiply(y_A, decision_vector_A)
    
        # compute the gradient
        grad_A = np.multiply((x_new_A.T.dot(y_new_A)), -1)
    
        # compute new vector of diagonal elements of tranformation matrix G
        S_new_A = np.add(S_A, np.square(grad_A))
    
        # compute new decision boundary
        w_new_A = np.subtract(w_A, np.multiply(grad_A, np.multiply(float(eta_A)/float(B_A), np.divide(1, np.sqrt(S_new_A)))))
    
        # take projection of decision boundary so that it does not exit the set 1/srqt(regularizer_A)
        w_new_proj_A = np.multiply(w_new_A, min(1, 1/(float(np.sqrt(regularizer_A)) *  float(np.linalg.norm(np.multiply(S_new_A, w_new_A))))))
    
        # increment the counter and put it in count_list
        count_A = count_A + 1
        count_list_A.append(count_A)
    
        # decision boundary is new decision boundary, put it in w_A_list
        w_A = w_new_proj_A
        w_A_list.append(w_A)
        
        # vector of diagonal elements of tranformation matrix G is the new vector of diagonal elements of G
        S_A = S_new_A
    
        # calculate training error and put it in training_error_list
        y_pred_A = x_A.dot(w_A)
    
        training_error_A = float(np.logical_xor((y_pred_A > 0).astype(int), (y_A > 0).astype(int)).sum())/float(B_A)
        training_error_list_A.append(training_error_A)

    print "\n\n AdaGrad ran successfully on the training data ...\n"

    end_time5 = time.time() - start_time5

    print "\n\n Time taken to train data using AdaGrad (in seconds) = ", end_time5
    
    print "\n\n\n Plotting Training Error v/s Number of Iterations for the trained model for AdaGrad ...\n"

    # plot training error v/s number of iterations
    plt.title('\n Training Error v/s Number of Iterations\n')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Training Error')
    plt.plot(count_list_A, training_error_list_A, color = '#ffe4e1')
    tck1 = interpolate.splrep(count_list_A, training_error_list_A, k = 3, s = 900)
    training_error_list_A_int = interpolate.splev(count_list_A, tck1, der = 0)
    plt.plot(count_list_A, training_error_list_A_int, color = 'red', label = 'AdaGrad')
    plt.legend()
    plt.show()
    
    print "\n\n\n Plotting Training Error v/s Number of Iterations for the trained model for PEGASOS and AdaGrad...\n"

    # plot training error v/s number of iterations
    plt.title('\n Training Error v/s Number of Iterations\n')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Training Error')
    plt.plot(count_list_P[::999], training_error_list_P_int[::999], color = 'blue', label = 'PEGASOS')
    plt.plot(count_list_A[::999], training_error_list_A_int[::999], color = 'red', label = 'AdaGrad')
    plt.legend()
    plt.show()

    return w_P_list, w_A_list, count_list_P, count_list_A

def testAccuracies(X_test_sparse, Y_test, w_P_list, w_A_list, count_list_P, count_list_A):
    
    '''
    Question 1.6

    Read in the test CSV dataset. Perform all steps in Question 1.1 and 1.2. An
    additional step required is that the sentiments in the test set are 0, 2 and 4.
    Convert the tweets with sentiment 2 to 4. Report test accuracy and a plot of 
    test error vs number of iterations for classfiers trained in Part 1.4 and Part 1.5.
    
    '''
    
    print "\n\n\n\n Making predictions on the test data and calculating the classification accuracies ..."

    test_error_P = [float(np.logical_xor((X_test_sparse.dot(w_P) > 0).astype(int), (Y_test.astype(int) > 0).astype(int)).sum())/float(Y_test.shape[0]) for w_P in w_P_list] 
    
    print "\n\nPercentage Accuracy for PEGASOS: " + str(round(100 - (test_error_P[-1] * 100), 3)) + "%"

    test_error_A = [float(np.logical_xor((X_test_sparse.dot(w_A) > 0).astype(int), (Y_test.astype(int) > 0).astype(int)).sum())/float(Y_test.shape[0]) for w_A in w_A_list]

    print "\n\nPercentage Accuracy for AdaGrad: " + str(round(100 - (test_error_A[-1] * 100), 3)) + "%"
    
    print "\n\n\n Plotting Test Error v/s Number of Iterations for the trained model for PEGASOS and AdaGrad...\n"
    
    # plot test error v/s number of iterations
    plt.title('\n Test Error v/s Number of Iterations\n')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Test Error')
    plt.plot(count_list_P, test_error_P, color = '#e0ffff')
    plt.plot(count_list_A, test_error_A, color = '#ffe4e1')
    tck1 = interpolate.splrep(count_list_P, test_error_P, k = 3, s = 900)
    test_error_P_int = interpolate.splev(count_list_P, tck1, der = 0)
    tck2 = interpolate.splrep(count_list_A, test_error_A, k = 3, s = 900)
    test_error_A_int = interpolate.splev(count_list_A, tck2, der = 0)
    plt.plot(count_list_P, test_error_P_int, color = 'blue', label = 'PEGASOS')
    plt.plot(count_list_A, test_error_A_int, color = 'red', label = 'AdaGrad')
    plt.legend()
    plt.show()
    
    return test_error_P, test_error_A

# **********************************************************************************************
#
#   function calls
#    
# **********************************************************************************************
    
# load training data
X_train, Y_train = loadTrainData('training.1600000.processed.noemoticon.csv')

# load test data
X_test, Y_test = loadTestData('testdata.manual.2009.06.14.csv')

# generate feature matrices of training and test data
X_train_sparse, X_test_sparse = createUnigramFeatureMatrixFromTweets(X_train, X_test)

# get decision 
w_P_list, w_A_list, count_list_P, count_list_A = runPegasosAndAdaGrad(X_train_sparse, Y_train)

# print test accuracy results
test_error_P, test_error_A = testAccuracies(X_test_sparse, Y_test, w_P_list, w_A_list, count_list_P, count_list_A)

end_time = time.time() - start_time1

print "\n\n\n\n Total time taken by program to run (in seconds) = ", end_time
print "\n\n"
