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

## Author
Ravindra Mina 
