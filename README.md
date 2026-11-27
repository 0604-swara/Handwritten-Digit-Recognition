
# Handwritten Digit Recognition

## 1. Project Overview
This project uses Machine Learning to recognize handwritten digits
from grayscale images of size 8x8 pixels.

## 2. Dataset Information
- Dataset: Scikit-learn Digits Dataset
- Total Images: 1797
- Image Dimensions: 8x8
- Features: 64
- Classes: 10 (Digits 0-9)
- Pixel Values: 0 to 16

## 3. Data Preprocessing
- Dataset divided into 80% training and 20% testing.
- Stratified splitting used.
- Random state: 42.
- StandardScaler used where required.
- Pipeline used to prevent data leakage.

## 4. Models Compared
1. Dummy Classifier
2. K-Nearest Neighbors
3. Logistic Regression

## 5. Cross Validation
- StratifiedKFold
- Number of folds: 5
- Shuffle: True
- Random state: 42

## 6. Selected Model
KNN

## 7. Final Test Performance

- Accuracy: 0.9639
- Macro Precision: 0.9646
- Macro Recall: 0.9637
- Macro F1-score: 0.9634

## 8. Error Analysis
A confusion matrix was generated to identify classification errors.
Misclassified images were visualized to understand recurring
confusion patterns between handwritten digits.

## 9. Prediction
The predict.py script:
- Accepts an 8x8 pixel matrix.
- Validates dimensions and pixel values.
- Converts the image into 64 features.
- Loads the saved model.
- Predicts the digit.
- Returns probabilities for all 10 classes.

## 10. Installation

Install dependencies:

pip install -r requirements.txt

## 11. Run Prediction

python predict.py example_input.json

## 12. Limitations
- The model is trained on 8x8 grayscale images.
- Real-world camera images require preprocessing.
- Handwriting variations may affect prediction accuracy.
- Images must be converted to the expected 8x8 pixel format.

## 13. Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
