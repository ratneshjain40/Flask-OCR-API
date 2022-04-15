import pandas
import string
import os
import re
import random
import pickle
import cv2
import spacy
import pytesseract

# ADD PATH HERE for Tesseract-OCR

def cleanText(txt):
    whitespace = string.whitespace
    punctuation = '!#$%&\'()*+:;<=>?[\\]^`{|}~'

    tableWhitepsace = str.maketrans('','',whitespace)
    tablePunctuation = str.maketrans('','',punctuation)
    
    text = str(txt)
    text = text.lower()
    removeWhitespace = text.translate(tableWhitepsace)
    removePunctuation = removeWhitespace.translate(tablePunctuation)
    return str(removePunctuation)


model_ner = spacy.load('output/model-best/')

def run_npl(image):
    textData = pytesseract.image_to_data(image)
    textData = list(map(lambda x:x.split('\t'), textData.split('\n')))
    df = pandas.DataFrame(textData[1:], columns=textData[0])

    df.dropna(inplace=True)
    df['text'] = df['text'].apply(cleanText)
    df = df.query('text != "" ')
    content = " ".join([w for w in df['text']])

    doc = model_ner(content)

    docjson = doc.to_json()
    docjson.keys()

    doc_text = docjson['text']

    dataframe_tokens = pandas.DataFrame(docjson['tokens'])
    dataframe_tokens['tokens'] = dataframe_tokens[['start', 'end']].apply(
        lambda x:doc_text[x[0]:x[1]], axis=1
    )

    right_table = pandas.DataFrame(docjson['ents'])[['start','label']]
    dataframe_tokens = pandas.merge(dataframe_tokens, right_table, how='left', on='start')
    dataframe_tokens.fillna('O', inplace=True)

    df['end'] = df['text'].apply(lambda x:len(x)+1).cumsum() - 1
    df['start'] = df[['text', 'end']].apply(lambda x:x[1] - len(x[0]), axis=1)
    dataframe_info = pandas.merge(df, dataframe_tokens[['start','tokens','label']], how='inner', on='start')

    bb_df = dataframe_info.query('label != "O"')
    img = image.copy()

    for x,y,w,h,label in bb_df[['left', 'top', 'width', 'height', 'label']].values:
        x=int(x)
        y=int(y)
        w=int(w)
        h=int(h)

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, str(label), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    return img, bb_df['text'].values.tolist()