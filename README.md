# **Amazon Food Review Sentiment Analysis in Python**
## Overview
This project aims to analyze the sentiment of Amazon food reviews using two different models: VADER (Valence Aware Dictionary and sEntiment Reasoner) and a pre-trained RoBERTa (Robustly Optimized BERT Pretraining Approach) model from the Hugging Face library. By comparing these models, it can be determined which one provides more accurate sentiment analysis for the given dataset.
## Dependencies and Data Loading
First, we import the necessary libraries and load the dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from tqdm.notebook import tqdm

plt.style.use('ggplot')

# Load dataset
df = pd.read_csv(r'/content/amazonFood.csv')
df = df.head(10000)
df['Id'] = df.index
df = df.reset_index(drop=True)
```

### Exploratory Data Analysis
We visualize the distribution of review scores in the dataset.
```python
score_counts = df['Score'].value_counts().sort_index()
plt.figure(figsize=(10,5))
sns.barplot(x=score_counts.index, y=score_counts.values)
plt.title('Count of reviews by Amazon star reviews')
plt.xlabel('Review stars')
plt.show()
```
### Sentiment Analysis with VADER
We use NLTK's VADER to analyze the sentiment of the reviews.
```python
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
  text = row['Text']
  myid = row['Id']
  res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
```

## Sentiment Analysis with RoBERTa
We use a pre-trained RoBERTa model for sentiment analysis.
```python
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def polarity_scores_roberta(example):
  encoded_text = tokenizer(example, return_tensors='pt')
  output = model(**encoded_text)
  scores = softmax(output[0][0].detach().numpy())
  scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
  }
  return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
  try:
    text = row['Text']
    myid = row['Id']
    vader_result = sia.polarity_scores(text)
    vader_result_rename = {f'vader_{key}': value for key, value in vader_result.items()}
    roberta_result = polarity_scores_roberta(text)
    both = {**vader_result_rename, **roberta_result}
    res[myid] = both
  except Exception as e:
    print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
```

## Comparison of Models
We compare the sentiment scores given by **VADER** and **RoBERTa**.
```python
sns.pairplot(data=results_df, vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'], hue='Score', palette='tab10')
plt.show()
```

## Conclusion
Both **VADER** and **RoBERTa** provide valuable insights into the sentiment of Amazon food reviews. However, based on the analysis, **RoBERTa** outperforms VADER in capturing the nuances of sentiment in the text. This is evident from its ability to better differentiate between positive, neutral, and negative sentiments across the reviews. The RoBERTa model's transformer architecture, which considers the context of words in a sentence, allows it to provide more accurate sentiment scores compared to the simpler bag-of-words approach used by VADER. Therefore, for more nuanced sentiment analysis, using of the RoBERTa model from the Hugging Face library is recommended.


For any questions or further assistance, please feel free to contact me.

Dhananjaya Mudunkotuwa  
dhananjayamudunkotuwa1998@gmail.com
