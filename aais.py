import pandas as pd
import os
import numpy as np
import tensorflow as tf
import pysbd
import re
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
from transformers import TextClassificationPipeline
import os

# we could check for existence first
data_location = os.environ.get("INPUT_DATA_DIR")
output_location = os.environ.get("OUTPUT_DIR")

def main():
    seg = pysbd.Segmenter(language="en", clean=False)
    model = TFDistilBertForSequenceClassification.from_pretrained("vangenugtenr/autobiographical_interview_scoring")
    tokenizer = AutoTokenizer.from_pretrained("vangenugtenr/autobiographical_interview_scoring")

    allDat = pd.read_csv(data_location)
    allDat.dropna(subset = ["participantID", "prompt", 'text'], inplace=True)

    list_of_dataframes = []
    for row in range(allDat.shape[0]):
        
        # access some general info about this narrative
        this_subID = allDat.iloc[row, allDat.columns.get_loc("participantID")]
        this_prompt = allDat.iloc[row, allDat.columns.get_loc("prompt")]
        narrative = allDat.iloc[row, allDat.columns.get_loc("text")]

        # store current row
        currentRow = allDat.iloc[[row], :] # if don't have brackets around row, not returned as Df, which is needed for merge

        # create new dataframe with each row a new sentence, and subID and prompt added
        segmented_sentences = seg.segment(narrative)
        sentences_df = pd.DataFrame(segmented_sentences, columns=['sentence'])
        sentences_df["participantID"] = this_subID
        sentences_df["prompt"] = this_prompt

        # create a new merged dataframe 
        merged_thisNarrative = pd.merge(currentRow, sentences_df, on=["participantID", "prompt"])

        list_of_dataframes.append(merged_thisNarrative)
    
    testData = pd.concat(list_of_dataframes)


    # now, make sure data are character
    testData.loc[:,'sentence'] = testData.loc[:,'sentence'].astype('str') 
    # create list of texts to classify (put in list format to encode texts)
    test_texts = []
    for row2 in range(testData.shape[0]):
        temp_test = testData.iloc[row2, testData.columns.get_loc("sentence")]
        temp_test = str(temp_test) # strip name of dataframe, then turn into string
        test_texts.append(temp_test)


    # encode text into something that bert can work with.
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings)
    ))

    # set up text classification pipeline using our model and tokenizer
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    # to keep amount of ram low, so that we can use the free version of
    # google colab, split classification up into batches rather than all
    # sentences at once.
    stored_test = []
    batch_size = 200  
    for i in range(0, len(test_texts), batch_size):
        stored_test.extend(pipe(test_texts[i:i+batch_size]))


    list_of_predictionDfs = []
    for row in range(len(stored_test)):
        thisTestLabels = pd.DataFrame(stored_test[row]) 
        thisTestLabels.index = thisTestLabels['label']
        thisTestLabels = thisTestLabels.drop('label', axis = 1)
        thisTestLabels = thisTestLabels.transpose()
        list_of_predictionDfs.append(thisTestLabels)
    predictionsDf = pd.concat(list_of_predictionDfs)


    predictionsDf['toplabel'] = predictionsDf.idxmax(axis=1)
    testData2 = pd.concat([testData.reset_index(drop=True), predictionsDf.reset_index(drop=True)], axis=1)


    testData2[['sentenceWordCount']] = 0
    for row in range(testData2.shape[0]):
        line = testData2.iloc[row, testData2.columns.get_loc("sentence")]
        count = len(re.findall(r'\w+', line))
        testData2.iloc[row, testData2.columns.get_loc("sentenceWordCount")] = count
    
    
    testData2[['numInt_preds']] = 0
    testData2[['numExt_preds']] = 0
    # now loop through each row and add in the counts
    for row in range(testData2.shape[0]):
        predictionType_thisIter = testData2.iloc[row, testData2.columns.get_loc("toplabel")]
        numTotalWords = testData2.iloc[row, testData2.columns.get_loc("sentenceWordCount")]

        internalLocation = testData2.columns.get_loc("numInt_preds")
        externalLocation = testData2.columns.get_loc("numExt_preds")
        
        if predictionType_thisIter == 'LABEL_0':
            testData2.iloc[row, externalLocation] = numTotalWords

        if predictionType_thisIter == 'LABEL_1':
            halfDetails = numTotalWords/2
            testData2.iloc[row, externalLocation] = halfDetails
            testData2.iloc[row, internalLocation] = halfDetails

        if predictionType_thisIter == 'LABEL_2':
            testData2.iloc[row, externalLocation] = numTotalWords/4
            testData2.iloc[row, internalLocation] = numTotalWords*(3/4)
                
        if predictionType_thisIter == 'LABEL_3':
            testData2.iloc[row, internalLocation] = numTotalWords
            
            
    test_write_out_subset = testData2.loc[:,["participantID","prompt", "text","numInt_preds", "numExt_preds", 'sentenceWordCount']]
    grouped = test_write_out_subset.groupby(by = ["participantID", "prompt"]).agg({'text': 'first', 
                                                'numInt_preds': 'sum', 
                                                'numExt_preds': 'sum',
                                                'sentenceWordCount': 'sum'})

    grouped.rename(columns = {"sentenceWordCount": "totalWordCount"}, 
            inplace = True)


    grouped.to_csv(output_location + '/automated_autobio_scored.csv') 
    return None

if __name__ == "__main__":
    main()