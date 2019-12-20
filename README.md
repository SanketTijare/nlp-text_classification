Text Classification
-----

This is a simple text classification problem tried to solve using NLP.

## Problem-Statment
-----

To classify the movie reviews by users into 2 classes 
	a. Positive 
	b. Negative

## Dataset-Overview
-----

Data set used is the "Large Movie Review Dataset v1.0", and can be downloaded from 'https://ai.stanford.edu/~amaas/data/sentiment/'

The core dataset contains 50,000 reviews split evenly into 25k train
and 25k test sets. The overall distribution of labels is balanced (25k
pos and 25k neg). We also include an additional 50,000 unlabeled
documents for unsupervised learning. 

There are two top-level directories [train/, test/] corresponding to
the training and test sets. Each contains [pos/, neg/] directories for
the reviews with binary labels positive and negative. Within these
directories, reviews are stored in text files named following the
convention [[id]_[rating].txt] where [id] is a unique id and [rating] is
the star rating for that review on a 1-10 scale. For example, the file
[test/pos/200_8.txt] is the text for a positive-labeled test set
example with unique id 200 and star rating 8/10 from IMDb. The
[train/unsup/] directory has 0 for all ratings because the ratings are
omitted for this portion of the dataset.


## Steps
-----

Below are the few basic high level steps that are followed in the moview classification.
- Load the data(corpus, both positive and negative) and converted it to a pandas dataframe
- Split data into Train and Test
- Use any NLP technique (TF-IDF or word counter to convert text to numbers)
- Train any Classifier 
- Evaluate the Classifier


