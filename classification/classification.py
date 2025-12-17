"""
Step 8: Analysis using Scikit-learn Classification part
Song Lyrics Classification: Top_Artists vs Low_Artists 
"""
#import all required ones
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import all required classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

#load data and preprocess
print("="*80)
print("SONG LYRICS CLASSIFICATION - SCIKIT-LEARN")
print("Classification Task: Top_Artists vs Low_Artists")
print("="*80)

# Load the CSV file using DataFrame
print("\n1. Loading UDAT features CSV file...")
df = pd.read_csv('output.csv', encoding='latin1')
print(f"   Original data shape: {df.shape}")
print(f"   Number of features: {df.shape[1] - 1}")  # Minus the Class column, just to see the features

# Display class distribution, checking balanced or not 
class_counts = df['Class'].value_counts()
print(f"\n   Class distribution:")
print(f"   - Top_Artists: {class_counts.get('Top_Artists', 0)}")
print(f"   - Low_Artists: {class_counts.get('Low_Artists', 0)}")

# Drop the Path column, if it is there
if 'Path' in df.columns:
    df = df.drop('Path', axis=1)
    print("\n2. Dropped 'Path' column")
else:
    print("\n2. No 'Path' column to drop")

# Shuffle the dataset to ensure random selection
print("\n3. Shuffling dataset...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("   Dataset shuffled with random_state=42")

# Splitting data into training and test sets

print("\n4. Splitting data into training and test sets...")

# Split using df.iloc(70 train, 30 percent test split)
train_size = int(0.7 * len(df))
train_df = df.iloc[:train_size]  
test_df = df.iloc[train_size:]   

print(f"   Training set: {train_df.shape[0]} samples")
print(f"   Test set: {test_df.shape[0]} samples")
print(f"   Split ratio: 70/30")

# separate the labels and features
print("\n5. Separating labels and features...")
# Separate the labels as a list
train_labels = train_df['Class'].tolist()
test_labels = test_df['Class'].tolist()

# Remove the Class column from samples
train_samples_df = train_df.drop('Class', axis=1)
test_samples_df = test_df.drop('Class', axis=1)

print(f"   Feature columns: {train_samples_df.shape[1]}")
# Converting data for Scikit-Learn
print("\n6. Converting data for scikit-learn...")

# Replace class names with 0 and 1
# 0 for "Low_Artists", 1 for "Top_Artists"
train_labels = [0 if label == 'Low_Artists' else 1 for label in train_labels]
test_labels = [0 if label == 'Low_Artists' else 1 for label in test_labels]

print("   Class encoding: Low_Artists=0, Top_Artists=1")

# Convert DataFrame to list of lists
train_samples = train_samples_df.values.tolist()
test_samples = test_samples_df.values.tolist()

print(f"   Training samples converted to list: {len(train_samples)} x {len(train_samples[0])}")
print(f"   Test samples converted to list: {len(test_samples)} x {len(test_samples[0])}")

# Handle NaN and infinite values
print("\n7. Handling NaN and infinite values...")
train_samples = [[0 if (pd.isna(x) or np.isinf(x)) else x for x in sample] for sample in train_samples]
test_samples = [[0 if (pd.isna(x) or np.isinf(x)) else x for x in sample] for sample in test_samples]
print("   NaN and infinite values replaced with 0")

# CLASSIFICATION USING MULTIPLE ALGORITHMS

print("\n" + "="*80)
print("CLASSIFICATION RESULTS")
print("="*80)

# Define classifiers to test (as specified in assignment)
classifiers = [
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('SVC', SVC()),
    ('NuSVC', NuSVC()),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('AdaBoost', AdaBoostClassifier(algorithm='SAMME', random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('GaussianNB', GaussianNB()),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    ('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis())
]

# Store results for table
results = []

# Test each classifier with the same code (as shown in assignment)
for name, clf in classifiers:
    try:
        # Train the classifier
        clf.fit(train_samples, train_labels)
        
        # Make predictions on test set
        predictions = clf.predict(test_samples)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        
        # Calculate metrics as specified in assignment
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        
        # Store results
        results.append({
            'Algorithm': name,
            'Classification Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        })
        
        print(f"\n{name}:")
        print(f"  Classification Accuracy: {accuracy:.16f}")
        print(f"  Sensitivity: {sensitivity:.16f}")
        print(f"  Specificity: {specificity:.16f}")
        
    except Exception as e:
        print(f"\nError with {name}: {str(e)[:100]}")
        results.append({
            'Algorithm': name,
            'Classification Accuracy': 0,
            'Sensitivity': 0,
            'Specificity': 0
        })

# DISPLAY RESULTS TABLE


print("\n" + "="*80)
print("RESULTS TABLE")
print("="*80)
print("-"*110)
print(f"{'Algorithm':<30} {'Classification Accuracy':<25} {'Sensitivity':<25} {'Specificity':<25}")
print("-"*110)

for result in results:
    print(f"{result['Algorithm']:<30} {result['Classification Accuracy']:<25.16f} "
          f"{result['Sensitivity']:<25.16f} {result['Specificity']:<25.16f}")

print("-"*110)


# SAVE RESULTS AND SUMMARY

# Save results to CSV for documentation
results_df = pd.DataFrame(results)
results_df.to_csv('classification_results.csv', index=False)
print(f"\nResults saved to: classification_results.csv")

# Find and display best performer
best_result = max(results, key=lambda x: x['Classification Accuracy'])
print(f"\n" + "="*80)
print("BEST PERFORMING CLASSIFIER")
print("="*80)
print(f"Algorithm: {best_result['Algorithm']}")
print(f"Classification Accuracy: {best_result['Classification Accuracy']:.4f}")
print(f"Sensitivity: {best_result['Sensitivity']:.4f}")
print(f"Specificity: {best_result['Specificity']:.4f}")


# CODE SUMMARY (AS REQUIRED IN ASSIGNMENT)

print("\n" + "="*80)
print("CODE USED (Required for Assignment Documentation):")
print("="*80)
print("""
The classification was performed using the following steps:
Trained and tested 10 different classifiers:
   - K-Nearest Neighbors
   - SVC
   - NuSVC
   - Decision Tree
   - Random Forest
   - AdaBoost
   - Gradient Boosting
   - GaussianNB
   - Linear Discriminant Analysis
   - Quadratic Discriminant Analysis,
Calculated metrics using confusion matrix:
    - Classification Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Sensitivity = TP / (TP + FN)
    - Specificity = TN / (TN + FP)
""")

print("="*80)
print("CLASSIFICATION ANALYSIS COMPLETE")
print("="*80)