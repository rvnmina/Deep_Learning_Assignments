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


# Assignment: Gradient Descent Based Optimization Algorithms

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


## Author
Ravindra Mina 
