**Slide 1: Objective**

* Build a high-performing image classification system without relying on deep learning, specifically CNNs, to demonstrate the potential of traditional machine learning when designed thoughtfully.
* Illustrate that effective feature engineering and model selection can lead to competitive results without the overhead of deep learning.
* Focus on delivering a solution that is not only accurate but also interpretable, resource-efficient, and easy to deploy across different environments.

**Slide 2: Dataset and Preprocessing**

* Dataset consists of 28x28 grayscale digit images stored in `.npy` format for efficient loading and manipulation.
* Pixel values are normalized to the \[0, 1] range, which is a standard preprocessing step that improves numerical stability and convergence during learning.
* An 80:20 train-validation split is used to allow performance evaluation while retaining sufficient data for training. The random seed ensures reproducibility.
* These steps ensure consistency, fair evaluation, and preparedness for downstream modeling.

**Slide 3: Feature Extraction Using HOG**

* With CNNs disallowed, Histogram of Oriented Gradients (HOG) is chosen for its ability to capture edge and shape information, which are critical for distinguishing digit forms.
* HOG transforms each image into a descriptor based on gradient orientations, effectively summarizing structural features.
* Parameters such as 9 orientations and 4x4 cell size are selected based on empirical best practices for 28x28 images.
* HOG excels at preserving the outline of digits, especially useful for differentiating similar digits like 3 vs 8 or 5 vs 6.

**Slide 4: Dimensionality Reduction Using UMAP**

* The extracted HOG vectors are high-dimensional, which can slow down learning and lead to overfitting.
* UMAP (Uniform Manifold Approximation and Projection) is selected to reduce dimensionality while maintaining the topological structure of data.
* StandardScaler is applied before UMAP to ensure features contribute equally—this is essential as UMAP is sensitive to feature scaling.
* PCA and UMAP plots are generated to visually verify cluster separation and determine if reduced features retain structure.

**Slide 5: Clustering – GMM vs KMeans**

* UMAP-reduced features are clustered using two techniques to explore unsupervised digit grouping.
* KMeans assumes spherical clusters and is computationally efficient, but struggles when digit shapes are not evenly distributed.
* GMM offers soft clustering with probabilistic outputs, enabling better modeling of overlapping or skewed clusters.
* Both are evaluated using Silhouette Score, Adjusted Rand Index (ARI), and Macro F1 Score to assess compactness, alignment with pseudo-labels, and balance across classes.
* GMM is chosen for its ability to model uncertainty and capture more realistic cluster boundaries.

**Slide 6: Cluster Labeling**

* Clusters from unsupervised learning do not carry semantic meaning, so a labeling step is introduced.
* Users either manually label clusters based on sampled images or use a predefined mapping for consistency.
* This process converts the clustering task into a semi-supervised classification problem.
* Sample visualization ensures clusters correspond to meaningful digit classes and improves label quality.

**Slide 7: SMOTE + Tomek Link Resampling**

* Post-labeling, some digit classes may be underrepresented due to uneven clustering.
* SMOTE (Synthetic Minority Over-sampling Technique) is used to generate synthetic examples for minority classes.
* Tomek Link removes borderline samples that cause class overlap, thus improving decision boundaries.
* This combined strategy improves classifier robustness and ensures class balance for fair training.

**Slide 8: Classifier Evaluation**

* Random Forest classifiers are trained using three different feature sets:

  * Raw flattened pixel values
  * Scaled HOG features
  * UMAP-reduced HOG features
* Each setup is evaluated using Adjusted Rand Index (ARI), Macro F1 Score, and overall accuracy.
* The UMAP-based classifier significantly outperforms others:

  * ARI: 0.9756
  * Macro F1 Score: 0.9889
* This confirms that dimensionality reduction helps in capturing the most discriminative aspects of digit shapes while also improving computational efficiency.

**Slide 9: Final Pipeline and Model Saving**

* Based on evaluation, the final pipeline uses HOG + UMAP + Random Forest.
* The following models and transformers are saved for reuse and deployment:

  * `hog_scaler.pkl`: for consistent preprocessing of HOG features.
  * `umap_model.pkl`: for dimensionality reduction.
  * `gmm_cluster_model.pkl`: for cluster assignment.
  * `cluster_to_digit_mapping.pkl`: for translating cluster IDs to digit labels.
  * `rf_umap_classifier.pkl`: final digit classifier.
* This modular saving strategy ensures that the entire pipeline can be reloaded and executed in production without retraining.

**Slide 10: Prediction Pipeline**

* A reusable function `predict_clusters()` encapsulates the end-to-end prediction flow.
* Accepts a new `.npy` file as input and performs:

  * HOG feature extraction
  * Feature scaling
  * UMAP projection
  * GMM-based clustering
  * Final classification using trained RF model
* Pseudo-labels from GMM are mapped to digits using the saved mapping.
* This pipeline supports scalable batch predictions with minimal configuration, and is fully portable.

**Slide 11: Conclusion**

* This project demonstrates that classical machine learning, when paired with strong feature engineering and dimensionality reduction, can achieve near-deep-learning accuracy.
* The HOG + UMAP + RF pipeline is compact, interpretable, and efficient, making it a great alternative when deep learning is not feasible.
* Every step in the pipeline is motivated by empirical best practices and justified through evaluation metrics.
* The end result is a modular and reproducible system suitable for deployment in real-world scenarios.
