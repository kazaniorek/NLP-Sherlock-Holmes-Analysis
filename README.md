# NLP Sherlock Holmes Analysis

This project applies modern Natural Language Processing (NLP) techniques to a collection of Sherlock Holmes stories. The aim is to demonstrate the use of deep learning and transformer-based models in summarisation, semantic search, and topic modelling for literary analysis.

## Features

### Task 1 – Summarisation

- Dataset: 67 Sherlock Holmes stories (.txt format)
- Preprocessing: Header removal, special character cleaning, word count analysis
- Models:
  - Baseline: LSTM Seq2Seq
  - T5 transformer (pre-trained and fine-tuned)
  - BART transformer (pre-trained, with longer input, and fine-tuned)
- Evaluation using ROUGE-1 and ROUGE-L metrics

### Task 3 & 4 – Semantic Search and Topic Modelling

- Embedding-based semantic search using Sentence Transformers (`all-MiniLM-L6-v2`)
- Storage and retrieval using ChromaDB
- Traditional TF-IDF comparison
- Topic modelling with Latent Dirichlet Allocation (LDA)
- Topic distribution visualisation (e.g. "Crime & Investigation", "Deduction", etc.)

## Technologies Used

- Python
- Hugging Face Transformers (T5, BART)
- TensorFlow / Keras
- Scikit-learn
- ChromaDB
- NLTK / spaCy
- LDA (via sklearn or gensim)

## File Structure

- `Notebook__NR1_Task1_Summaries_30030295.ipynb` – summarisation experiments
- `Notebook_NR2_Task3_and_Task4_30030295.ipynb` – semantic search & topic modelling

## How to Run

1. Install required packages:
   ```bash
   pip install transformers sentence-transformers chromadb sklearn pandas matplotlib
   ```

2. Open notebooks in Jupyter or Colab and run cells sequentially.

3. Make sure the `data/` folder contains the Sherlock Holmes text files in `.txt` format.

## Author

Mariusz Sołtycz

---

This project was completed as part of my BSc Artificial Intelligence degree (Module: Deep Learning – CS4S772). It demonstrates the real-world application of NLP and XAI techniques to large-scale, unstructured text data.
