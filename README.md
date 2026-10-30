# Deep Learning NLP (NLP for ML and NLP for DL)

Jupyter notebooks that build up NLP step by step, from text preprocessing to word embeddings, LSTMs and BERT.

| Notebook | Topic | Data |
|---|---|---|
| `NLP_Text_Preprocessing.ipynb` | Tokenizing, stemming, lemmatization, stopwords, bag of words | Text in the notebook |
| `Assignment_Practise_Text_preprocessing.ipynb` | Practice: preprocessing, BoW and TF-IDF | Text in the notebook |
| `Word_Embedding_NLP.ipynb` | One-hot encoding, padding, and a Keras Embedding layer | Text in the notebook |
| `Deep_Learning_Word2Vec.ipynb` | Pre-trained Word2Vec with gensim | Downloaded automatically |
| `NLP_Word2vec_And_AvgWord2vec.ipynb` | Spam classifier with BoW, TF-IDF and average Word2Vec | `smsspamcollection/SMSSpamCollection`* |
| `Assignment_IMDB_Dataset_Word2Vec_and_AvgWord2Vec.ipynb` | Movie review sentiment with Word2Vec | `IMDB Dataset.csv`* |
| `FakeNewsClassifierUsingLSTM.ipynb` | Fake news detection with an LSTM | `train.csv`* |
| `Custom_Sentiment_Analysis.ipynb` | Fine-tune DistilBERT for spam detection | `SMSSpamCollection`* |
| `Transformers_and_Bert.ipynb` | Hugging Face pipelines, tokenizers and models | Downloaded automatically |
| `Bert.ipynb` | BERT embeddings from TensorFlow Hub | Downloaded automatically |

\* Not included in the repository. See [Datasets](#datasets).

## How to run (step by step)

You need **Python 3.10 – 3.12** ([download](https://www.python.org/downloads/)). These versions are the safest for TensorFlow. No API key is needed.

> **Windows:** `tensorflow-text` (used only by `Bert.ipynb`) has no Windows build. Run `Bert.ipynb` in Google Colab. If `pip install` fails on that package, remove the `tensorflow-text` line from `requirements.txt` and run it again.

> **Easiest option:** use Google Colab (free GPU). It already has most packages installed. See below.

**1. Download the project**

```bash
git clone https://github.com/amitmohanty022/Deep-Learning-NLP-for-ML-NLP-for-DL.git
cd Deep-Learning-NLP-for-ML-NLP-for-DL
```

**2. Create a virtual environment and activate it**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install the requirements** (TensorFlow and PyTorch are large, so this takes a few minutes)

```bash
pip install -r requirements.txt
```

**4. Add the data files** that the notebook you want needs (see [Datasets](#datasets)). Put them in this folder.

**5. Start Jupyter**

```bash
jupyter notebook
```

Open a notebook and choose **Run → Run All Cells**.

### Running in Google Colab instead

1. Go to https://colab.research.google.com → **File → Upload notebook**.
2. For GPU: **Runtime → Change runtime type → T4 GPU**.
3. Upload the data file with the 📁 **Files** panel on the left.
4. **Runtime → Run all**. The first cell of each TensorFlow notebook installs what it needs.

## Datasets

| File | Where to get it |
|---|---|
| `SMSSpamCollection` | UCI SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection. Unzip it. `NLP_Word2vec_And_AvgWord2vec.ipynb` expects it inside a `smsspamcollection/` folder. |
| `IMDB Dataset.csv` | Kaggle: "IMDB Dataset of 50K Movie Reviews" |
| `train.csv` | Kaggle "Fake News" competition: https://www.kaggle.com/c/fake-news/data |

The notebooks read data files by name only (for example `'IMDB Dataset.csv'`), so the same code works locally and in Colab.

## Updates (2026)
- TensorFlow 2.16+ ships Keras 3, which removed `one_hot`, `Embedding(input_length=...)` and the Hugging Face TF integration. The TensorFlow notebooks now start with `TF_USE_LEGACY_KERAS=1` and use `tf-keras` (Keras 2), so the original code runs unchanged.
- Hugging Face Transformers 5 dropped TensorFlow, so `transformers` is pinned to the 4.57 series.
- `TFTrainer` was removed from Transformers. `Custom_Sentiment_Analysis.ipynb` now trains with Keras `compile()`/`fit()` and computes real predictions for the confusion matrix.
