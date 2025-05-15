
---

### **Slide 1: Project Title**

**Clustering and Classification of Unlabelled Handwritten Images**
*Using Deep Feature Extraction and Pseudo-Labelling*

This project is focused on enabling intelligent classification of image data in situations where manual labelling is either unavailable or infeasible. We explore how deep learning and clustering methods can work in tandem to unlock structure and meaning from raw image pixels.

---

### **Slide 2: Objective**

The goal of this project is to build a complete and reusable machine learning pipeline that can:

* Interpret unlabelled handwritten digit images and uncover meaningful patterns.
* Convert unsupervised learning output (clusters) into structured, pseudo-labelled data.
* Train a classifier on this generated data to make predictions on new, unseen inputs.
* Save and reuse all trained models, so that the system is ready for deployment or integration in production workflows.

The underlying challenge here is to simulate the benefits of supervised learning, even when labels are not provided.

---

### **Slide 3: Dataset Summary**

We work with a dataset of 60,000 handwritten grayscale images, each of size 28×28 pixels. The images are stored in a NumPy `.npy` format, which is efficient to load and avoids the need for image decoding or transformation.

Key characteristics:

* Shape: `(60000, 1, 28, 28)`
* No associated ground truth labels
* Ideal use case for unsupervised learning methods like clustering

This dataset poses a real-world challenge—learning structure purely from raw pixel data without any prior annotations.

---

### **Slide 4: Preprocessing Method**

To make the raw image data usable for deep learning:

* **Normalization** is applied by dividing pixel values by 255. This ensures all values lie between 0 and 1, which prevents unstable gradients and speeds up learning.
* **Reshaping** is performed to match the format expected by convolutional layers in Keras: `(height, width, channels)` = `(28, 28, 1)`.
* The dataset is split into training and validation sets (80% and 20%). A fixed seed (`random_state=42`) is used to ensure that model results can be reproduced exactly.

Each of these steps contributes to training stability, consistency, and compatibility with deep learning frameworks.

---

### **Slide 5: Feature Extraction Using CNN**

Instead of using raw pixels directly, we build a CNN to extract meaningful visual features:

* The network includes three convolutional layers that detect shapes, curves, and edges.
* Batch normalization helps speed up training and reduces sensitivity to initialization.
* MaxPooling reduces spatial dimensions, retaining dominant features.
* Dropout layers reduce the risk of overfitting, especially important when training without ground truth labels.

The CNN is not trained to classify digits but rather to transform raw pixel data into feature vectors that encode rich semantic information. These features will later be used for clustering.

---

### **Slide 6: Using UMAP for Feature Reduction**

Once CNN features are extracted, they are still high-dimensional. To enable effective clustering:

* We reduce them to 10 dimensions using UMAP (Uniform Manifold Approximation and Projection).
* UMAP is a powerful dimensionality reduction tool that preserves both local neighborhoods and global structure of the data.
* Compared to PCA or t-SNE, UMAP is faster, scales better with data size, and yields more meaningful clusterable spaces.

UMAP-transformed data becomes the input to clustering algorithms and also feeds into 2D visualizations for human inspection.

---

### **Slide 7: Clustering Strategy**

To identify structure in the unlabelled dataset:

* We use **KMeans**, which partitions data into k distinct groups based on distance from centroids.
* We also use **GMM**, which assumes the data is generated from a mixture of Gaussian distributions.

Both methods are initialized to form 10 clusters, reflecting the assumption that the digits 0 through 9 are represented in the dataset.

Each image is assigned a cluster ID, which is used as a temporary label (pseudo-label). These pseudo-labels act as training targets for the next step.

---

### **Slide 8: Cluster Validation via Visualization**

To verify that our clustering process grouped similar images together:

* We visualize samples from each cluster and inspect their visual patterns.
* For example, Cluster 3 might contain images that resemble '5', while Cluster 7 might show variations of '2'.

This visual inspection validates that the learned features and clustering logic are capturing real-world structure and not just arbitrary pixel similarities. It also reassures us that pseudo-labels are meaningful.

---

### **Slide 9: Addressing Imbalance in Pseudo-Labels**

Initial cluster distributions are not uniform—some clusters have far fewer images than others. This is problematic for training supervised classifiers.

To resolve this:

* We apply **SMOTE** to generate synthetic samples for smaller clusters.
* We apply **Tomek Links** to remove samples that are ambiguous and lie near cluster boundaries.

Together, this creates a dataset where all clusters are represented more evenly, improving model generalization and reducing classifier bias.

---

### **Slide 10: Supervised Classification**

Using the rebalanced pseudo-labelled dataset:

* We train a **Random Forest Classifier** using the flattened image data.
* Random Forest is chosen for its robustness to noise, ability to handle multi-class data, and resistance to overfitting.
* This classifier learns to associate pixel patterns with pseudo-labels.

The aim here is not to model the original digit labels (which we don’t have) but to learn to predict consistent cluster membership for new data points.

---

### **Slide 11: Evaluation Strategy**

To understand how well the model performs, we compute several metrics:

* **Silhouette Score**: Quantifies how tightly grouped the clusters are.
* **Adjusted Rand Score**: Measures agreement between clustering assignments and classifier predictions.
* **Macro F1 Score**: Provides a balanced accuracy measure across all clusters.
* **Confusion Matrix**: Visualizes which clusters the classifier finds hard to distinguish.

Together, these metrics give us a multi-dimensional view of how well the system is learning, clustering, and classifying.

---

### **Slide 12: Model Persistence**

To make the pipeline usable beyond the notebook:

* We save the CNN model (`cnn_feature_extractor.keras`) to preserve the feature extraction process.
* The UMAP model (`.pkl`) is saved to apply the same transformation on new data.
* Clustering models (KMeans, GMM) and the classifier (Random Forest) are stored for future inference.

This makes the system portable, shareable, and usable on new unlabelled image datasets without retraining from scratch.

---

### **Slide 13: Unified Prediction Function**

A function called `predict_clusters()` wraps the entire pipeline into a single callable method:

* It loads new `.npy` data
* Applies the CNN, UMAP, clustering models, and classifier
* Returns predicted cluster IDs and classifier outputs

This function enables one-step batch inference and allows the entire workflow to be reused without manual intervention.

---

### **Slide 14: Conclusion**

This project shows that it is possible to create a full classification system without access to labelled data:

* CNN-based feature extraction enables images to be represented in a meaningful numeric form.
* Clustering techniques help uncover structure and grouping in unlabelled data.
* Pseudo-labels allow training a classifier to predict new images accurately.
* The pipeline is designed to be modular, reusable, and efficient—suitable for production or research use.

This approach opens possibilities for many domains where labelled data is expensive or unavailable, such as satellite imagery, medical scans, and historical archives.

---

