# Mild Cognitive Disorder Markers (Polish) - PoC

A classical machine learning pipeline for detecting linguistic markers associated with Mild Cognitive Impairment (MCI) in Polish language descriptions of the "Cookie Theft" clinical picture task.

## What it does

* **Analyzes Polish text** for specific linguistic features: Type-Token Ratio (TTR), average sentence length, and pronoun-to-noun ratios.
* **Classifies input** as either `Healthy (0)` or `MCI (1)` using a Random Forest classifier.
* **Prioritizes Explainability:** Unlike black-box deep learning models, this pipeline extracts transparent Gini importance scores, allowing clinicians to see exactly which linguistic markers drove the classification.

## Stack

| Component | Detail |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **NLP Pipeline** | spaCy (`pl_core_news_md`) — Polish lemmatization & POS tagging |
| **ML Framework** | scikit-learn (RandomForestClassifier) |
| **Data Handling** | Pandas & NumPy |
| **Visualization** | Seaborn & Matplotlib |

## How to run

1. Open `classic-ml-mci-markers-pl.ipynb` in Google Colab.
2. Select runtime: `Runtime → Change runtime type → T4 GPU` (not strictly required for this CPU-light classical pipeline, but good practice for consistency).
3. Run cells in order (Steps 1–5) to generate synthetic data, extract features, and train the model.

## Pipeline

| Step | Description |
| :--- | :--- |
| 1 | Setup environment & load Polish spaCy models |
| 2 | Synthetic data generation (100 unique samples) |
| 3 | Feature engineering (Clinical markers: TTR, Anomia indicators) |
| 4 | Exploratory Data Analysis (Visualizing class separation) |
| 5 | Model training, Evaluation, & Feature Importance extraction |

## Clinical Rationale

In early-stage cognitive decline, patients often exhibit **anomia** (word-finding difficulty). This results in a decreased use of specific nouns and an increased reliance on pronouns (e.g., "to", "tamto"). Our pipeline specifically quantifies this **Pronoun-to-Noun Ratio**, providing a statistically backed, explainable marker for clinical screening.

## Example output
