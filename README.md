# Banknote Authentication Using SVM

## Project Overview
This project demonstrates the process of authenticating banknotes using a Support Vector Machine (SVM) classifier. The dataset contains features extracted from wavelet-transformed images of genuine and forged banknotes. The goal is to build a model that accurately classifies banknotes as authentic or forged based on these features.

## Dataset
**Source**: UCI Machine Learning Repository (ID 267)  
**Features**:
- `variance`: Variance of wavelet-transformed images.
- `skewness`: Skewness of wavelet-transformed images.
- `curtosis`: Kurtosis of wavelet-transformed images.
- `entropy`: Entropy of images.  
**Target**: `class` (0 = authentic, 1 = forged).  

The dataset contains **1,372 instances** and is publicly available via the `ucimlrepo` Python package.

## Key Steps
### 1. Data Loading and Exploration
- Import libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
- Fetch and load the dataset.
- Perform exploratory data analysis (EDA):
  - Descriptive statistics.
  - Univariate analysis (distributions, outliers, skewness).
  - Visualizations (histograms, KDE plots, boxplots).

### 2. Preprocessing
- Split data into training and testing sets (`train_test_split`).
- Standardize features using `StandardScaler`.

### 3. Model Training
- Train an SVM classifier (`SVC` from `scikit-learn`).
- Optimize hyperparameters if needed (not explicitly shown in the notebook).

### 4. Evaluation
- Calculate accuracy using `accuracy_score`.
- Generate a confusion matrix (`confusion_matrix`).
- Visualize decision boundaries (if applicable).

### 5. Results
- The model achieves high accuracy (exact value depends on runtime results).
- Confusion matrix highlights true positives/negatives and misclassifications.

## Dependencies
- Python 3.x
- Libraries:  
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo
  ```

## How to Run
1. Clone the repository or download the Jupyter notebook (`Project11_Banknote_Authentication_Using_SVM.ipynb`).
2. Install required libraries (see **Dependencies**).
3. Execute the notebook cells sequentially to:
   - Load and explore the data.
   - Preprocess features.
   - Train and evaluate the SVM model.
   - Visualize results.

## Code Structure
- **Data Loading**: Fetches data from the UCI repository.
- **EDA**: Analyzes distributions, uniqueness, and statistical summaries.
- **Preprocessing**: Splits data and scales features.
- **Modeling**: Implements SVM and evaluates performance.
- **Visualization**: Includes plots for data exploration and model evaluation.

## Insights
- The SVM model performs well on this dataset due to clear feature separability.
- Features like `variance` and `entropy` show distinct distributions for authentic vs. forged banknotes.
- Data preprocessing (scaling) is critical for SVM performance.

## License
This project uses the [UCI Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication), which is publicly available for research purposes.  


---
