# Food_categories_deep_learning
Food-101 Image Classification: Building a Robust Deep Learning Model for Food Recognition

# Custom CNN for Food Image Classification

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains the complete PyTorch implementation for building, training, and evaluating a custom Convolutional Neural Network (CNN) to classify a subset of 10 food classes from the Food-101 dataset. The project covers the entire machine learning workflow, from data preparation and augmentation to model training, hyperparameter tuning, and final evaluation.

## Key Features

* **End-to-End Pipeline**: A single, well-commented script that handles the entire process.
* **Custom CNN Architecture**: A CNN built from scratch using `torch.nn`, demonstrating foundational principles.
* **Comprehensive Data Augmentation**: A robust pipeline using `torchvision.transforms` to improve model generalization, including rotations, crops, flips, color jitter, and shearing.
* **Systematic Evaluation**: Implements a 70/15/15 split for training, validation, and testing, and uses Early Stopping to prevent overfitting.
* **In-depth Analysis**: Generates a classification report and a confusion matrix to provide deep insights into the final model's performance.

## Dataset

This project uses a subset of 10 classes from the public **Food-101** dataset, https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Food101.html. The chosen classes are:

* `apple_pie`
* `baby_back_ribs`
* `baklava`
* `beef_carpaccio`
* `beef_tartare`
* `beet_salad`
* `beignets`
* `bibimbap`
* `bread_pudding`
* `breakfast_burrito`

The data is expected to be organized into `train`, `validation`, and `test` directories, with each directory containing subfolders for each of the 10 classes.

Let´s do a initial Data Exploration:

<img width="2489" height="2244" alt="image" src="https://github.com/user-attachments/assets/a538a1b5-5ac6-4e8d-a544-de8209dbe66b" />


## Methodology

The approach follows standard best practices for training a deep learning model for computer vision:

1.  **Data Loading**: Uses `torchvision.datasets.ImageFolder` to load the pre-organized data.
2.  **Data Augmentation**: The training set undergoes a series of random transformations to create a robust model. The validation and test sets are only resized and normalized to ensure a consistent evaluation.
3.  **Model Architecture**: A custom CNN with 3 convolutional blocks followed by 2 fully-connected layers. Regularization techniques like `Dropout` and `L2 Regularization (Weight Decay)` are employed.
4.  **Hyperparameter Tuning**: The project was tuned for key hyperparameters like learning rate and weight decay to optimize performance.
5.  **Training**: The model was trained using the Adam optimizer and Cross-Entropy Loss, with Early Stopping to automatically save the best model based on validation loss.

## Getting Started

To run this project yourself, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-link]
    cd [your-repo-name]
    ```

2.  **Set up your environment:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Organize your data:**
    Create a main data folder (e.g., `./data/`) and inside it, create three folders: `train`, `validation`, and `test`. In each of these, create 10 subfolders with the class names listed above and place the corresponding images inside.

4.  **Run the training script:**
    Update the `base_data_dir` variable in the script to point to your main data folder, then run:
    ```bash
    python your_script_name.py
    ```

## Results
Training Architecture 1: Baseline Convolutional Neural Network (CNN)

Our initial architecture was a custom-built Convolutional Neural Network (CNN). This model was designed to serve as a fundamental baseline to establish benchmark performance on our dataset. The architecture consisted of a standard sequence of convolutional layers, ReLU activations, and max-pooling layers to extract features, followed by fully-connected layers for classification.

<img width="678" height="393" alt="image" src="https://github.com/user-attachments/assets/c5c0b7f7-11d0-4901-9a22-dbd4d192e917" />

Training architecture 2:
For our second training architecture, we selected ResNet-18 to enhance model performance. This well-established convolutional neural network (CNN) utilizes residual connections to enable effective training of deeper models. This deeper architecture allows for the extraction of more complex features from the data, which is expected to lead to a significant improvement in classification accuracy.

<img width="442" height="405" alt="image" src="https://github.com/user-attachments/assets/ce3912c1-2942-42fc-b95e-2fbff1ce9e44" />


<img width="543" height="422" alt="image" src="https://github.com/user-attachments/assets/7602f1cc-1afc-489d-8d0e-055f2887f3b6" />

## Conclusion

The project started with a baseline model that severely overfit the data, peaking at only 50% accuracy. An initial fix using regularization stabilized the model but did not improve performance.

The major breakthrough came from abandoning the simple model and using transfer learning combined with systematic hyperparameter tuning. This new approach dramatically increased performance to 72.8% accuracy. The future goal is to further refine this model to reach 80-90% accuracy.

<img width="763" height="407" alt="image" src="https://github.com/user-attachments/assets/5fce0021-14ba-4b4f-9eb8-69c70aa37553" />


After a series of hyperparameter tuning experiments, our best model achieved the following performance on the held-out test set:

## Future Work

* **Implement Transfer Learning**: Compare the custom CNN's performance against a pre-trained model like ResNet18 to see the impact of transfer learning.
* **Advanced Augmentation**: Explore more advanced data augmentation techniques.
* **Automated Tuning**: Use a library like Optuna or Ray Tune to perform a more exhaustive hyperparameter search.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Thanks to the creators of the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
