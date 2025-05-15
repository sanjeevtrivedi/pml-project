Self-Supervised Text Classification Using Unlabelled Data
This document presents a complete walkthrough of the end-to-end pipeline built for self-supervised text classification using unlabelled raw text. The goal of this project was not only to label unclassified documents but to do so in a manner that is scalable, explainable, and generalizable to future, unseen data.

Problem Context
In many practical scenarios, especially in business environments, we are faced with large repositories of unstructured and unlabelled text data. These could be customer reviews, internal notes, emails, or research archives. Manually labelling this data is infeasible. Traditional supervised learning fails in such a scenario because it requires labeled examples for training.
The goal of this project was to derive meaningful clusters from raw text, assign each document to a discovered topic, and then train a predictive model that could classify new documents using this self-labelled data.

Initial Data Assessment
The dataset comprised 1500 raw text documents in a .pkl format. As the first step, a thorough audit of the dataset was performed to eliminate any noise that could interfere with modeling:
* Empty documents and duplicates were removed. These do not contribute meaningfully to topic modeling and introduce noise.
* A document-length distribution was plotted to understand whether certain entries were significantly shorter or longer, which could affect cluster placement.
* A WordCloud and frequency histogram were generated to visualize the dominant terms in the corpus. This gave us an early intuition about the nature of the topics embedded in the data.
By the end of this phase, the dataset was confirmed to be clean and structurally ready for preprocessing.

Text Cleaning and Normalization
Preprocessing is critical for any NLP task. The following steps were used to standardize the text:
* Lowercasing: Converted all text to lowercase to avoid treating the same word in different cases as separate tokens.
* Punctuation and Symbol Removal: Special characters were stripped to focus on lexical meaning.
* Stopword Removal: Words that don’t contribute to topic differentiation (e.g., "the", "is", "and") were excluded.
* Lemmatization: Words were reduced to their base form using WordNetLemmatizer. This ensured that words like "running" and "ran" were treated as the same feature.
These steps ensured that the input passed into the vectorization layer was as consistent and semantically meaningful as possible.

Feature Representation Using TF-IDF + LSA
TF-IDF was used to convert text into numerical vectors. This method helps emphasize uncommon, but topic-relevant, terms while down-weighting frequently occurring but uninformative ones.
Following this, dimensionality reduction was applied using Latent Semantic Analysis (LSA) via Truncated SVD:
* This reduced the TF-IDF matrix to 100 components.
* More than 25% of the variance was retained in these components, confirming that the representation was informative and compact.
Choosing LSA over raw TF-IDF vectors helped reduce sparsity, making clustering more effective and efficient.

Clustering and Pseudo-Label Generation
Two clustering methods were tested:
* KMeans: Popular, scalable, and well-suited to compact embeddings
* Agglomerative Clustering: Hierarchical, useful for exploratory validation
Both were evaluated using:
* Silhouette Score: Measures cohesion within clusters
* Davies-Bouldin Index: Measures separation between clusters
KMeans was chosen due to better separability and faster convergence.
Each document was assigned a cluster label (pseudo-label) based on the KMeans output. These labels served as the ground truth for classifier training.

Interpreting Cluster Semantics
To interpret each cluster, we:
* Computed average TF-IDF scores for each cluster.
* Extracted top keywords to understand the theme.
* Assigned a single-word human-readable label (e.g., "Religion", "Hardware") to each cluster.
This step transformed numeric labels into domain-aware topics. These names were used in classification reports, plots, and confusion matrices, improving interpretability across the pipeline.

Imbalance Detection and Balancing Strategy
Upon inspecting cluster distributions, some were underrepresented. Training classifiers on imbalanced data would result in biased models skewed toward dominant clusters.
To correct this, SMOTE (Synthetic Minority Oversampling Technique) was used. This generated new training samples in underrepresented clusters by interpolating between existing points.
This choice helped retain all samples from dominant clusters while equalizing representation during classifier training.

Supervised Classification using Pseudo-Labels
Three supervised models were trained:
* Logistic Regression: Fast and baseline interpretable model
* Random Forest: Handles non-linearity and performs internal feature selection
* SVM: Performs well with high-dimensional sparse data
Each model was tuned using GridSearchCV with macro F1-score as the scoring metric.
Macro F1-score was chosen because it:
* Evaluates performance for each class independently
* Does not favor majority classes
* Reflects classifier performance across all discovered topics
Models were evaluated using classification reports and confusion matrices, with all outputs mapped to the semantic cluster names.

Comparing Imbalanced vs SMOTE-Enhanced Training
To measure the effect of SMOTE, each classifier was trained on both the original imbalanced data and the SMOTE-balanced data.
A bar chart of macro F1-scores revealed consistent improvement (2–4%) in all classifiers after applying SMOTE.
This confirmed that data balancing significantly enhanced generalization, especially for minority clusters.

Explainability via Feature Importance
Feature importances were extracted from the Random Forest model. The top contributing terms for each class aligned well with the manually interpreted cluster themes.
This added a layer of explainability to the model, making it more transparent and trustworthy.

Benchmarking with SBERT and BERT
To test whether semantic embeddings could outperform traditional methods, we evaluated:
* SBERT: Sentence embeddings
* BERT: Token embeddings with contextual information
Results:
* Low Silhouette Scores (~0.04–0.05)
* Lower macro F1-scores (~0.81)
* Less distinct clusters observed in UMAP plots
In this case, TF-IDF + LSA outperformed SBERT and BERT, due to the clearly separable topical nature of the data.

Final Model Selection and Export
The best performing model (TF-IDF + Logistic Regression) was selected based on macro F1-score.
Artifacts saved:
* Logistic Regression model (final_best_model.pkl)
* TF-IDF vectorizer (final_vectorizer.pkl)
* SVD transformer (final_svd.pkl)
These were bundled into an inference pipeline for future predictions.

Inference Pipeline and Deployment Readiness
The pipeline supports three forms of input:
* Raw text string list
* .pkl file of documents
* .csv file with a text column
Each prediction includes:
* Predicted cluster name
* Model confidence score
This ensures real-time and batch predictions are both supported and interpretable.

Sample Inference
"The moon mission was a great success by NASA."
# Output: ('Space', Confidence: 0.94)

Conclusion
This project successfully demonstrates:
* The transformation of unlabelled raw text into structured, labelled output using self-supervised techniques
* The effective integration of clustering and classification into a unified pipeline
* That traditional TF-IDF methods offer superior performance over deep learning embeddings in this topic-based domain
* The importance of combining explainability, class balancing, and a modular inference design to produce a deployable and interpretable ML system