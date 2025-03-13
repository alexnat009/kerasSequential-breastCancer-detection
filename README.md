# Breast Cancer Detection Using Neural Networks  

## Project Overview  
This project implements a **Neural Network Model** to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** based on diagnostic measurements. The dataset consists of various **cellular features** extracted from digitized breast cancer images.  

## Model Architecture  
The neural network consists of:  
- **Input Layer** (128 neurons, ReLU activation)  
- **Dropout Layer** (50% dropout)  
- **Hidden Layer** (64 neurons, ReLU activation)  
- **Dropout Layer** (50% dropout)  
- **Output Layer** (1 neuron, Sigmoid activation for binary classification)  

The model is trained using the **Adam optimizer** with **binary cross-entropy loss**.  

## Dataset  
The dataset is stored in the following file:  
```
dataset/data.csv
```
This dataset contains **diagnostic features** extracted from breast cancer images.  

## nstallation & Setup  
1) Clone the repository:  
```bash
git clone https://github.com/alexnat009/kerasSequential-breastCancer-detection.git  
cd kerasSequential-breastCancer-detection 
```  
2) Install dependencies:  
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn  
```  
3) Run the model training:  
```bash
python main.py  
```  

## Model Training & Evaluation  
- The model is trained for **100 epochs**  
- Training and validation accuracy/loss are plotted for analysis  
- A **confusion matrix** is generated to evaluate model performance  

## 📈 Visualization  
The script generates **accuracy & loss curves** and **confusion matrix heatmap** to monitor model performance

## License  
This project is **open-source** under the [MIT License](LICENSE).  
