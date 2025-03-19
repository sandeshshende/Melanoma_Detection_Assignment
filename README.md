# Melanoma Skin Cancer Detection

## Abstract

The objective of this project is to develop an automated classification system using image processing techniques to accurately identify and classify skin cancer, specifically melanoma, from dermoscopic images of skin lesions. Melanoma, the deadliest form of skin cancer, requires early detection to improve treatment outcomes significantly. Currently, dermatologists capture dermoscopic images using high-resolution cameras, achieving diagnostic accuracies between 65% and 80% without technical assistance. With additional analysis by oncologists and dermoscopic evaluations, this accuracy improves to around 75% to 84%. By implementing advanced image processing and classification techniques, this project aims to enhance the accuracy and efficiency of melanoma detection, thereby supporting timely diagnosis and better clinical decisions.

## Problem statement

The problem statement of this project is to develop a CNN-based model capable of accurately detecting melanoma, a life-threatening type of skin cancer responsible for 75% of skin cancer-related deaths. Early detection of melanoma is critical to improving survival rates, and an automated solution that can evaluate dermoscopic images and alert dermatologists about the potential presence of melanoma can greatly reduce the manual effort involved in diagnosis. By leveraging deep learning techniques, particularly Convolutional Neural Networks (CNNs), the system will analyze skin lesion images and provide a reliable classification, enabling faster and more accurate detection, ultimately aiding in timely intervention and treatment.

## Table of Contents

- [General Info](#general-information)
- [Model Architecture](#model-architecture)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Acknowledgements](#acknowledgements)
- [Collaborators](#collaborators)

<!-- You can include any other section that is pertinent to your problem -->

## General Information

The dataset used for this project consists of 2,357 images representing both malignant and benign oncological conditions, obtained from the International Skin Imaging Collaboration (ISIC). The images have been categorized according to the classification provided by ISIC, ensuring that each subset contains an equal number of images. This balanced dataset is essential for training the CNN model effectively, preventing class imbalance issues, and ensuring that the model can accurately distinguish between malignant and benign conditions.

![datasetgraph](https://github.com/user-attachments/assets/883f1bc3-f098-42a8-af42-6d7c130f9483)


To address the challenge of class imbalance, the Augmentor Python package (Augmentor Documentation) was utilized to augment the dataset. This process involved generating additional samples for all classes by applying various transformations such as rotation, flipping, zooming, and scaling. As a result, the dataset was balanced, ensuring that no class had insufficient representation, which helps improve the model’s performance and prevents bias toward the majority class.

## Pictorial representation of skin types

![skincancertypes](https://github.com/user-attachments/assets/f3bbae95-2ab3-4fd5-910b-8005e2f383f1)


The aim of this task is to develop a classification model that assigns a specific class label to each type of skin cancer based on the analysis of dermoscopic images.

## Model Architecture

The break down of the final provided CNN architecture step by step:

1. **Data Augmentation**: The `augmentation_data` applies random transformations (rotation, scaling, flipping) to increase the training data diversity, improving model generalization.

2. **Normalization**: The `Rescaling(1./255)` normalizes pixel values to a range between 0 and 1, stabilizing training and speeding up convergence.

3. **Convolutional Layers**: Three convolutional layers are added sequentially using the `Conv2D` function. Each convolutional layer is followed by a rectified linear unit (ReLU) activation function, which introduces non-linearity into the model. The `padding='same'` argument ensures that the spatial dimensions of the feature maps remain the same after convolution. The number within each `Conv2D` layer (16, 32, 64) represents the number of filters or kernels used in each layer, determining the depth of the feature maps.

4. **Pooling Layers**: The `MaxPooling2D` reduces the spatial dimensions of feature maps after each convolution, lowering computational complexity and preventing overfitting.

5. **Dropout Layer**: A dropout layer (`Dropout`) randomly drops neurons to reduce overfitting.

6. **Flatten Layer**: The `Flatten` layer converts 2D feature maps into a 1D vector for input into dense layers.

7. **Fully Connected Layers**: Two fully connected (dense) layers (`Dense`) are added with ReLU activation functions. The first dense layer consists of 128 neurons, and the second dense layer outputs the final classification probabilities for each class label.

8. **Output Layer**: The output layer predicts class labels, with the number of neurons equal to `target_labels`

9. **Model Compilation**: The model is compiled using the Adam optimizer (`optimizer='adam'`) and the Sparse Categorical Crossentropy loss function (`loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`), which is suitable for multi-class classification problems. Additionally, accuracy is chosen as the evaluation metric (`metrics=['accuracy']`).

10. **Training**: The model is trained using the `fit` method with the specified number of epochs (`epochs=50`). The `ModelCheckpoint` and `EarlyStopping` callbacks are employed to monitor the validation accuracy during training. The `ModelCheckpoint` callback saves the model with the best validation accuracy, while the `EarlyStopping` callback stops training if the validation accuracy does not improve for a specified number of epochs (patience=5 in this case). These callbacks help prevent overfitting and ensure that the model converges to the best possible solution.


## Model Evaluation

![Model Evaluation](https://github.com/user-attachments/assets/c4eb3e2f-d72b-4062-8b12-a8fec5ea4d3c)


## Technologies Used

- [Python](https://www.python.org/) - version 3.11.11
- [Matplotlib](https://matplotlib.org/) - version 3.10.0
- [Numpy](https://numpy.org/) - version 2.0.2
- [Pandas](https://pandas.pydata.org/) - version 2.2.2
- [Seaborn](https://seaborn.pydata.org/) - version 0.13.2
- [Tensorflow](https://www.tensorflow.org/) - version 2.18.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements

- UpGrad tutorials on Convolution Neural Networks (CNNs) on the learning platform

## Collaborators

Created by 
- [@Sandesh_Shende](https://github.com/sandeshshende)
