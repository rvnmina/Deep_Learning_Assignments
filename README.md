# Assignment 1: MP Neuron Based Logic Circuits

## Objective
To design and implement basic arithmetic logic circuits using the McCulloch–Pitts (MP) neuron model.

## Implemented Circuits
- Half Adder using minimum number of MP neurons
- Full Adder using two Half Adders and an OR gate

## Concepts Used
- McCulloch–Pitts neuron model
- Linear threshold units
- Inhibitory inputs
- Boolean logic implementation using neurons

## Files
- `MP_Neuron_Half_Full_Adder.ipynb` – Complete implementation and testing

## Results
All circuits were tested on all possible input combinations and the outputs match the theoretical truth tables.




# Assignment 2: Perceptron Learning Algorithm on Iris Dataset

## Objective
To implement the Perceptron learning algorithm from scratch and analyze its performance on a multiclass classification problem using the Iris dataset.

## Implemented Tasks
- Loaded the Iris dataset from scikit-learn
- Split the dataset into training and test sets (70%–30%)
- Implemented the Perceptron algorithm from scratch
- Used One-vs-Rest strategy for multiclass classification
- Trained each classifier until misclassifications stopped reducing
- Plotted misclassification trends across epochs
- Evaluated performance using accuracy, precision, and recall
- Compared results with scikit-learn’s Perceptron implementation
- Visualized feature-wise class separability

## Concepts Used
- Perceptron learning algorithm
- Linear threshold units
- One-vs-Rest multiclass classification
- Misclassification-based convergence
- Performance evaluation metrics
- Data visualization for separability analysis

## Files
- `Perceptron_Iris_Assignment.ipynb` – Complete implementation, training, evaluation, and visualizations

## Results
The custom Perceptron achieved performance comparable to scikit-learn’s implementation. The Setosa class was linearly separable from the other classes, while Versicolor and Virginica showed overlapping regions, indicating partial linear separability of the dataset.


# Assignment 3: Deep Learning for Regression and Model Comparison

## Objective

To design, train, and evaluate neural network models for a regression task using the Communities and Crime dataset, and to compare NumPy-based and PyTorch-based implementations.

## Implemented Tasks

- Dataset loading and preprocessing  
- Handling missing values and feature normalization  
- Train, validation, and test dataset splitting  
- Neural network training for regression  
- Hyperparameter (δ) selection using validation R² score  
- Test set evaluation and prediction visualization  
- PyTorch implementation with identical architecture  
- Performance comparison between NumPy-based and PyTorch models  

## Concepts Used

- Feedforward Neural Networks (MLP)  
- Regression using neural networks  
- Feature scaling using StandardScaler  
- Mean Squared Error (MSE) and Huber loss  
- R² score for model evaluation  
- Hyperparameter tuning and model selection  
- Overfitting and generalization analysis  
- Comparison of deep learning frameworks  

## Files

- `Assignment_3_Deep_Learning_Regression.ipynb` – Complete implementation and analysis  
- `best_model_weights.h5` – Saved weights of the best-performing model  
- `README.md` – Assignment overview  

## Results

The neural network models were trained and evaluated successfully on the Communities and Crime dataset.  
The optimal value of δ was selected based on validation R² score.  
Both NumPy-based and PyTorch implementations achieved comparable test-set R² scores, indicating consistent and reliable performance.

## Conclusion

This assignment demonstrates the effective application of deep learning techniques for regression problems.  
The comparison between NumPy-based and PyTorch implementations highlights consistency in model behavior across frameworks while reinforcing core deep learning concepts.


# Assignment 4: Gradient Descent Based Optimization Algorithms

---

## Objective

To design, implement, and analyze gradient descent based optimization algorithms for training a fully connected neural network, and to compare custom implementations with PyTorch’s built-in optimizers.

---

## Tasks Performed

- Loaded and preprocessed the Bike Sharing Dataset  
- Handled missing values and duplicate records  
- Performed feature selection and one-hot encoding  
- Split data into training, validation, and test sets  
- Applied feature scaling and target transformation  
- Designed a fully connected neural network for regression  
- Implemented optimization algorithms from scratch  
- Trained the same model using PyTorch optimizers  
- Evaluated model performance using standard regression metrics  

---

## Implemented Optimization Algorithms

- Gradient Descent with Momentum (from scratch)  
- Adam Optimizer (from scratch)  
- Adam Optimizer using PyTorch  

---

## Neural Network Architecture

- **Input Layer:** Dataset-dependent feature dimension  
- **Hidden Layers:**
  - 256 units – ReLU activation  
  - 128 units – ReLU activation  
  - 64 units – ReLU activation  
  - 32 units – ReLU activation  
- **Output Layer:**
  - 1 unit – Linear activation (regression task)  

---

## Evaluation Metrics

- Mean Squared Error (MSE)  
- Coefficient of Determination (R² Score)  

---

## Results

- The custom implementation of Adam achieved good convergence and stable training.  
- PyTorch’s Adam optimizer demonstrated faster convergence and improved generalization.  
- The best-performing model achieved:
  - **Test R² ≈ 0.85**, explaining approximately **85% of the variance** in bike rental demand.  
- Scatter plots of predicted vs. true values show strong alignment along the diagonal, indicating excellent predictive performance.

---

## Concepts Used

- Gradient Descent Optimization  
- Momentum-Based Optimization  
- Adam Optimizer  
- Backpropagation  
- Feature Scaling and Target Transformation  
- Neural Networks for Regression  
- Model Evaluation and Comparison  

---

## Files

- `Gradient_Descent_Optimization.ipynb`  
  Complete implementation including data preprocessing, model training, optimization algorithms, evaluation, and visualizations.

---

## Conclusion

This assignment highlights the effectiveness of gradient descent based optimization techniques for training neural networks. The comparison between from-scratch implementations and PyTorch’s Adam optimizer demonstrates the advantages of optimized learning strategies in achieving better convergence and generalization.


# Assignment 5: Regularization Techniques for Neural Networks

---

## Objective

To design and train a fully connected neural network using PyTorch to predict the superconducting critical temperature.  
The assignment focuses on understanding the effect of Batch Normalization and batch size on model performance and generalization.

---

## Implemented Tasks

- Data preprocessing and feature scaling using StandardScaler  
- Conversion of dataset into PyTorch tensors and DataLoaders  
- Design of a feed-forward neural network with three hidden layers  
- Custom Batch Normalization implementation from scratch  
- Model training using multiple batch sizes  
- Test set evaluation using Mean Squared Error (MSE) and R² score  
- Batch size vs performance analysis  
- Prediction scatter plot visualization  
- PyTorch Batch Normalization implementation and comparison

---

## Model Architecture

- Input Layer: 81 features  
- Hidden Layer 1: 256 neurons + BatchNorm + ReLU  
- Hidden Layer 2: 128 neurons + BatchNorm + ReLU  
- Hidden Layer 3: 64 neurons + BatchNorm + ReLU  
- Output Layer: 1 neuron (Regression output)

Loss Function:
- Mean Squared Error (MSE)

Optimizer:
- Adam Optimizer

---

## Concepts Used

- Fully Connected Neural Networks
- Batch Normalization (Custom and PyTorch)
- Gradient Descent Optimization
- Model Generalization
- Regression Performance Metrics (MSE, R²)
- Data Normalization and Scaling

---

## Files

- `DL_Lab5.ipynb` – Complete implementation and experiments  
- `superconductivty+data.zip` – Dataset used for training and testing

---

## Results

- Models were trained using batch sizes: `{4, 8, 16, 32, 64, 128, 256}`  
- Medium batch sizes (32–64) achieved the best R² performance.  
- Custom Batch Normalization improved training stability.  
- PyTorch BatchNorm implementation showed strong generalization on the test set.

All experiments were implemented strictly using **PyTorch** without TensorFlow.

---

## Visualizations

- Training vs Validation Loss Curves
- Batch Size vs R² Performance Plot
- Prediction Scatter Plots (Predicted vs Ground Truth)

---
# Assignment 6: Object Localization using CNN (NumPy & PyTorch)

---

## Objective

To design and train a Convolutional Neural Network (CNN) from scratch using NumPy and compare it with a PyTorch implementation for object localization on the Oxford-IIIT Pets dataset.

This assignment focuses on bounding box regression, CNN architecture design, and evaluation using IoU (Intersection over Union).

---

## Implemented Tasks

- Loaded Oxford-IIIT Pets dataset
- Generated bounding boxes from segmentation masks
- Normalized bounding box coordinates
- Split dataset into training and validation sets (80-20)
- Implemented CNN from scratch using NumPy
- Designed Smooth L1 (Huber) loss
- Trained NumPy model using SGD with momentum
- Implemented same architecture in PyTorch
- Trained PyTorch model
- Evaluated using IoU metric
- Compared NumPy vs PyTorch performance
- Visualized predictions vs ground truth

---

## Model Architecture

### Input
- Image size: **64 × 64 × 3**

### Convolutional Layers
- Conv1 → 64 filters + BatchNorm + ReLU + MaxPool  
- Conv2 → 128 filters + BatchNorm + ReLU + MaxPool  
- Conv3 → 256 filters + BatchNorm + ReLU + MaxPool  
- Conv4 → 512 filters + BatchNorm + ReLU + MaxPool  

### Fully Connected Layers
- FC1: 8192 → 512 + ReLU + Dropout  
- FC2: 512 → 256 + ReLU + Dropout  
- Output: 256 → 4 (Bounding Box) + Sigmoid  

---

## Loss Function

- **Smooth L1 Loss (Huber Loss)**
- Robust for bounding box regression

---

## Optimizer

### NumPy Model
- SGD with Momentum
- Learning rate decay

### PyTorch Model
- SGD Optimizer (momentum = 0.9)

---

## Concepts Used

- Convolutional Neural Networks (CNN)
- Bounding Box Regression
- Batch Normalization
- Dropout Regularization
- Gradient Descent
- Smooth L1 Loss
- Intersection over Union (IoU)
- NumPy vs PyTorch comparison

---

## Files

- `assignment6.ipynb` — Complete implementation  
- `numpy_cnn_weights.npz` — NumPy model weights  
- `pytorch_bbox_model.pth` — PyTorch model weights  

---

## Results

- Trained for **20 epochs**
- Stable convergence observed

### Performance (Approx)

- **Mean IoU:** ~0.63 – 0.67  
- **Median IoU:** ~0.65 – 0.69  
- **IoU > 0.5:** ~85%+  
- **IoU > 0.3:** ~97%+  

---

## Observations

- PyTorch model performs slightly better  
- NumPy model helps understand backpropagation deeply  
- Good localization performance achieved  
- Predictions closely match ground truth  

---

## Visualization

- 🟩 Ground Truth  
- 🟥 NumPy CNN  
- 🟦 PyTorch CNN  

---

## Conclusion

This assignment demonstrates complete object localization pipeline using CNNs.

# Assignment — CNN & Adversarial Attacks

## Overview
This project implements a Convolutional Neural Network (CNN) for image classification and evaluates its robustness using adversarial attacks such as **DeepFool**. The assignment also compares a custom implementation with the `torchattacks` library.

---

## Objectives
- Build and train a CNN model
- Optimize using Adam optimizer
- Evaluate model performance
- Implement DeepFool adversarial attack
- Compare custom vs library-based attacks
- Visualize results

---

## Model Architecture
- Conv2D + BatchNorm + ReLU layers
- Residual/Skip connections
- Adaptive Average Pooling
- Final classifier layer

**Total Parameters:** ~4.4 Million

---

## Training Details
- Optimizer: Adam
- Learning Rate: 1e-3
- Scheduler: Cosine Annealing
- Epochs: 30

### Training Results
- Final Train Accuracy: **~97%**
- Final Validation Accuracy: **~96%**

✔ Smooth convergence observed  
✔ No major overfitting  

---

## Performance Evaluation

### Test Accuracy
- **~96% overall accuracy**

### Classification Metrics
- High precision, recall, and F1-score across all classes
- Balanced performance across categories

---

## Confusion Matrix
- Strong diagonal dominance
- Minimal misclassification
- Model performs consistently across classes

---

## DeepFool Attack (Custom Implementation)

### Results:
- Successfully fooled most correctly classified samples
- **Attack Success Rate: ~100%**
- Average perturbation: Low (imperceptible noise)

Insight:
> Model is highly vulnerable to adversarial perturbations.

---

## DeepFool using torchattacks

### Results:
- **Success Rate: ~93–95%**
- Slightly lower than custom implementation

Reason:
- Uses fixed number of iterations
- Less aggressive than custom approach

---

## Comparison

| Method                  | Success Rate |
|------------------------|-------------|
| Custom DeepFool        | ~100%       |
| torchattacks DeepFool  | ~93–95%     |

Difference: ~5–7%

### Conclusion:
Both implementations follow the same DeepFool algorithm.  
Differences arise due to:
- Iteration limits  
- Numerical precision  
- Implementation strategy  

---

## Visualization
- Loss vs Epoch graph
- Accuracy vs Epoch graph
- DeepFool comparison bar chart

---

## Key Learnings
- CNNs achieve high accuracy but are vulnerable to adversarial attacks  
- Small perturbations can significantly affect predictions  
- Attack implementations vary in effectiveness  

---

## Tech Stack
- Python  
- PyTorch 
- NumPy  
- Matplotlib  
- torchattacks  

---

## Project Structure

It highlights:
- Manual deep learning implementation (NumPy)
- Framework-based optimization (PyTorch)
- Real-world computer vision application

---
# Assignment 7: Human Activity Recognition using GRU (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## Overview

This project implements **Human Activity Recognition (HAR)** using deep learning on the **UCI HAR Dataset**.

Two models were developed:
- GRU model built from scratch
- GRU model using PyTorch (`nn.GRU`)

The objective is to classify human activities such as walking, sitting, and standing.

---

## Dataset

- UCI Human Activity Recognition Dataset
- 30 subjects
- 561 features
- 6 activity classes:
  - WALK
  - WALK_UPSTAIRS
  - WALK_DOWNSTAIRS
  - SITTING
  - STANDING
  - LAYING

---

## Preprocessing

- Subject-wise split:
  - Train: 20 subjects
  - Validation: 5 subjects
  - Test: 5 subjects
- Feature standardization (Z-score)
- Reshaping features into sequence format for GRU

---

## Models

### 1. GRU from Scratch
- Hidden Size: 64
- Layers: 2
- Dropout: 0.4

### 2. PyTorch GRU
- Implemented using `torch.nn.GRU`
- Same architecture for fair comparison

---

## Training Details

- Loss Function: CrossEntropyLoss
- Optimizer: AdamW
- Learning Rate: 1e-4
- LR Scheduler: ReduceLROnPlateau
- Gradient Clipping
- Early Stopping
- Best Model Saving

---

## Results

### From-Scratch GRU
- Accuracy: **93.71%**

### PyTorch GRU
- Accuracy: **91.80%**

---

## Comparison

| Model | Accuracy |
|------|--------|
| From-Scratch GRU | **93.71%** |
| PyTorch GRU | 91.80% |

The custom GRU model performs better and shows improved generalization.

---

## Observations

- High accuracy across all classes
- Minor confusion between:
  - SITTING and STANDING
  - WALK and WALK_UPSTAIRS

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn

---

## Author
Ravindra Mina 
