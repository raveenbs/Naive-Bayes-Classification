
########  Naive Bayes Classification  ##########

###################################################
# Step 1: Set the Working Directory
###################################################
setwd("~/raveen")

###################################################
# Step 2: Read in and Examine the Data
###################################################
sms <- read.csv("sms_spam.csv", stringsAsFactors=F)
View(sms)

table(sms$type)
?prop.table
round(prop.table(table(sms$type))*100, digits = 1)

###################################################
# Step 3: Assign type as a factor
###################################################
# Type column as factor
sms$type = factor(sms$type)

###################################################
# Step 4: Build a corpus (collection of documents)
###################################################
# tm package to build a corpus
library(tm)

# build a corpus, which is a collection of documents, from the texts
?VectorSource
?VCorpus
sms_corpus = VCorpus(VectorSource(sms$text))
lapply(sms_corpus[1:3],as.character)

###################################################
# Step 5: Preprocess the data in Corpus
###################################################
# clean up the data
corpus_clean = tm_map(sms_corpus, tolower)                    # convert to lower case
corpus_clean = tm_map(corpus_clean, removeNumbers)            # remove digits
corpus_clean = tm_map(corpus_clean, removeWords, stopwords()) # and but or you etc
corpus_clean = tm_map(corpus_clean, removePunctuation)        # No punctuation
corpus_clean = tm_map(corpus_clean, stripWhitespace)          # reduces w/s to 1
inspect(corpus_clean[1:3])

###################################################
# Step 6: Creating a spare matrix from Corpus
###################################################
# Creating a sparse matrix comprising:
# the columns are the union of words in our corpus
# the rows correspond to each text message
# the cells are the number of times each word is seen

corpus_clean = tm_map(corpus_clean, PlainTextDocument) 
dtm = DocumentTermMatrix(corpus_clean)
str(dtm)

###################################################
# Step 7: Split data into train and test
###################################################
set.seed(123)

# split the raw data:
sample <- sample.int(n = nrow(sms), size = floor(.75*nrow(sms)), replace = F)
sms.train <- sms[sample, ]
sms.test  <- sms[-sample, ]

# then split the document-term matrix
dtm.train <- dtm[sample, ]
dtm.test  <- dtm[-sample, ]

# and finally the corpus
corpus.train = corpus_clean[sample]
corpus.test  = corpus_clean[-sample]

# let's just assert that our split is reasonable: raw data should have about 87% ham
# in both training and test sets:
round(prop.table(table(sms.train$type))*100)

round(prop.table(table(sms.test$type))*100)


###################################################
# Step 8: Selecting high frequency words
###################################################

ncol(dtm.train)
ncol(dtm.test)
# DTMs have more than 7000 columns 
# To get a managable matrix 
# eliminate words which appear in less than 5 SMS messages 

freq_terms = findFreqTerms(dtm.train, 5)
reduced_dtm.train = DocumentTermMatrix(corpus.train, list(dictionary=freq_terms))
reduced_dtm.test =  DocumentTermMatrix(corpus.test, list(dictionary=freq_terms))

# Check reduced features
ncol(reduced_dtm.train)
ncol(reduced_dtm.test)

###################################################
# Step 9: Converting numberic values in dtm to character factors
###################################################
# Naive Bayes Classification works on factors, 
# Converting DTM Numerics to factor
convert_counts = function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels=c("No", "Yes"))
  return (x)
}

# apply() allows us to work either with rows or columns of a matrix.
# MARGIN = 1 is for rows, and 2 for columns
reduced_dtm.train = apply(reduced_dtm.train, MARGIN=2, convert_counts)
reduced_dtm.test  = apply(reduced_dtm.test, MARGIN=2, convert_counts)

###################################################
# Step 10: Run Naive Bayes Classification
###################################################
library(e1071)
# store our model in sms_classifier
?naiveBayes
sms_classifier = naiveBayes(reduced_dtm.train, sms.train$type)
sms_classifier$tables[1:5]
# Predict using the classifier (may take a minute to complete)
sms_test.predicted = predict(sms_classifier,
                             reduced_dtm.test)

###################################################
# Step 11: Crosstable for accuracy
###################################################
# use CrossTable() from gmodels 
# CrossTable will be used to build the confusion matrix 
# to check the accuracy of the model
library(gmodels)
CrossTable(sms_test.predicted,
           sms.test$type,
           prop.chisq = FALSE, # as before
           prop.t     = FALSE, # eliminate cell proprtions
           dnn        = c("predicted", "actual"))


###################################################
# Step 12: Improve classification with laplace smoothing
###################################################
# We need to reduce the 15 emails that are actually ham 
# but are classified as spam as they can be critical
# Using laplace smoothing
sms_classifier_l = naiveBayes(reduced_dtm.train, sms.train$type, laplace = 0.2)
sms_classifier_l$tables[1:5]
sms_test.predicted_l = predict(sms_classifier_l,
                             reduced_dtm.test)


CrossTable(sms_test.predicted_l,
           sms.test$type,
           prop.chisq = FALSE, # as before
           prop.t     = FALSE, # eliminate cell proprtions
           dnn        = c("predicted", "actual"))

###################################################
# Step 13: Comparison of errors
###################################################
# comparing the errors with and without laplace smoothing
# to confirm that our 4 remaining hams classified 
# as spam were part of the original 6 misclassifications
sms.test$pred = sms_test.predicted
Error_test <- subset(sms.test,type == "ham" & pred == "spam")
View(Error_test)

sms.test$pred = sms_test.predicted_l
Error_test_l <- subset(sms.test,type == "ham" & pred == "spam")
View(Error_test_l)
