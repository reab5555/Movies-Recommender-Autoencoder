# Netflix Movie Recommendation System using Autoencoder

This project implements a movie recommendation system using an Autoencoder to predict and recommend movies to users. The dataset used is a subset of the Netflix Prize dataset, which includes user ratings for movies.   
   
Due to the computational requirements, only a quarter of the dataset is used for training.   

## Table of Contents

- [Introduction](#introduction)
- [Collaborative Filtering](#collaborative-filtering)
- [Autoencoder Models](#autoencoder-models)
- [Dataset](#dataset)
- [Model Architecture and Hyperparameters](#model-architecture-and-hyperparameters)
- [Training and Validation](#training-and-validation)
- [Results](#results)
- [Prediction and Recommendation](#prediction-and-recommendation)
- [References](#references)

## Introduction

This project aims to build a movie recommendation system based on user ratings using an Autoencoder model. The primary goal is to predict and recommend movies that users are likely to enjoy based on their past ratings and the ratings of similar users.

## Collaborative Filtering

Collaborative Filtering (CF) is a technique used in recommendation systems where the preferences of users are predicted based on the preferences of other users. CF systems can be user-based, where similar users are identified, or item-based, where similar items (movies) are identified. In this project, we use a user-based CF approach.

## Autoencoder Models

An Autoencoder is a type of artificial neural network used to learn efficient codings of input data. The network consists of two main parts:
- **Encoder**: Compresses the input into a latent-space representation.
- **Decoder**: Reconstructs the input from the latent space.

By training the Autoencoder on user ratings, we can learn a compact representation of users' preferences and use it to predict ratings for movies that the users haven't seen.

## Dataset

The dataset used is a subset of the Netflix Prize dataset. This dataset includes user ratings for various movies. The ratings range from 1 to 5, and users have not rated all movies. We use the following data preprocessing steps:
1. Loading the dataset.
2. Creating a sparse user-movie matrix.
3. Splitting the data into training (80%) and testing (20%) sets.

### Dataset Information:
- Number of users: 470,758
- Number of movies: 4,499
- Total number of ratings: 24,053,764
- Sparsity: 0.0114

## Model Architecture and Hyperparameters

The Autoencoder model consists of the following layers:

### Encoder:
- Linear layer with 512 units, ReLU activation, Dropout
- Linear layer with 256 units, ReLU activation, Dropout
- Linear layer with 64 units, ReLU activation

### Decoder:
- Linear layer with 256 units, ReLU activation, Dropout
- Linear layer with 512 units, ReLU activation, Dropout
- Linear layer with input dimension units, ReLU activation

### Hyperparameters:
- **Encoding Dimension**: 64
- **Dropout Rate**: 0.1
- **Learning Rate**: 0.0001
- **Weight Decay**: 1e-5
- **Batch Size**: 128
- **Epochs**: 30
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with L2 regularization

## Training and Validation

The model is trained for 30 epochs with the following results:

| Epoch | Training Loss | Training RMSE | Validation Loss | Validation RMSE |
|-------|---------------|---------------|-----------------|-----------------|
| 1     | 1.7422        | 1.3199        | 1.0592          | 1.0292          |
| 2     | 0.9587        | 0.9791        | 0.8509          | 0.9225          |
| 3     | 0.8671        | 0.9312        | 0.8389          | 0.9159          |
| 4     | 0.8383        | 0.9156        | 0.8114          | 0.9008          |
| 5     | 0.8154        | 0.9030        | 0.8018          | 0.8954          |
| 6     | 0.8031        | 0.8961        | 0.7890          | 0.8882          |
| 7     | 0.7852        | 0.8861        | 0.7770          | 0.8815          |
| 8     | 0.7717        | 0.8785        | 0.7681          | 0.8764          |
| 9     | 0.7603        | 0.8720        | 0.7577          | 0.8704          |
| 10    | 0.7578        | 0.8705        | 0.7723          | 0.8788          |
| 11    | 0.7412        | 0.8610        | 0.7562          | 0.8696          |
| 12    | 0.7335        | 0.8565        | 0.7423          | 0.8616          |
| 13    | 0.7300        | 0.8544        | 0.7417          | 0.8612          |
| 14    | 0.7199        | 0.8484        | 0.7355          | 0.8576          |
| 15    | 0.7147        | 0.8454        | 0.7297          | 0.8542          |
| 16    | 0.7065        | 0.8405        | 0.7273          | 0.8528          |
| 17    | 0.7026        | 0.8382        | 0.7245          | 0.8512          |
| 18    | 0.6972        | 0.8350        | 0.7221          | 0.8498          |
| 19    | 0.6953        | 0.8338        | 0.7183          | 0.8475          |
| 20    | 0.6895        | 0.8304        | 0.7204          | 0.8488          |
| 21    | 0.6854        | 0.8279        | 0.7158          | 0.8460          |
| 22    | 0.6820        | 0.8258        | 0.7124          | 0.8440          |
| 23    | 0.6815        | 0.8255        | 0.7099          | 0.8425          |
| 24    | 0.6750        | 0.8216        | 0.7063          | 0.8404          |
| 25    | 0.6714        | 0.8194        | 0.7030          | 0.8384          |
| 26    | 0.6679        | 0.8173        | 0.6997          | 0.8365          |
| 27    | 0.6653        | 0.8157        | 0.6988          | 0.8360          |
| 28    | 0.6628        | 0.8141        | 0.6999          | 0.8366          |
| 29    | 0.6607        | 0.8128        | 0.6974          | 0.8351          |
| 30    | 0.6570        | 0.8106        | 0.6972          | 0.8350          |

## Results

The model was trained for 30 epochs, achieving the following results:

- **Best Training Loss**: 0.6570
- **Best Training RMSE**: 0.8106
- **Best Validation Loss**: 0.6972
- **Best Validation RMSE**: 0.8350

## Prediction and Recommendation

A separate script is used for predicting or recommending 10 movies for a new user who has rated only some of the movies. The recommendation is based on what the user has already rated and what similar users have rated. Since this is a CF system, the recommendations are generated based on the similarities between users.

## References

1. Netflix Prize Dataset. [Link to dataset](https://www.netflixprize.com/)
2. PyTorch Documentation. [Link to documentation](https://pytorch.org/docs/stable/index.html)

Feel free to explore and contribute to this project. Happy coding!
