import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.1):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if x.is_sparse:
            x = x.to_dense()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_model(model_path, input_dim, encoding_dim):
    model = Autoencoder(input_dim, encoding_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def create_user_vector(user_ratings, movie_to_id, num_movies):
    user_vector = np.zeros(num_movies)
    for movie_title, rating in user_ratings.items():
        if movie_title in movie_to_id:
            movie_id = movie_to_id[movie_title]
            user_vector[movie_id] = rating
    return user_vector


def get_top_n_recommendations(model, user_vector, id_to_movie, n=10):
    with torch.no_grad():
        user_tensor = torch.FloatTensor(user_vector).unsqueeze(0)
        predicted_ratings = model(user_tensor).squeeze(0).numpy()

    # Scale predictions to [1, 5] range
    predicted_ratings = np.clip(predicted_ratings, 1, 5)

    unrated_movies = np.where(user_vector == 0)[0]
    top_n_indices = unrated_movies[np.argsort(predicted_ratings[unrated_movies])[-n:][::-1]]
    top_n_ratings = predicted_ratings[top_n_indices]

    recommendations = []
    for idx, rating in zip(top_n_indices, top_n_ratings):
        if idx in id_to_movie:
            movie_title = id_to_movie[idx]
            recommendations.append((movie_title, rating))

    return recommendations


def main():
    # Load the new user's ratings
    new_user_df = pd.read_csv('test_user_recommend_ratings.csv')

    # Load movie metadata
    movies_df = pd.read_csv('netflix_ratings_with_titles.csv')

    # Create mappings between movie titles and IDs
    movie_to_id = dict(zip(movies_df['movie'], movies_df['movie_id']))
    id_to_movie = dict(zip(movies_df['movie_id'], movies_df['movie']))

    # Create a dictionary of the user's ratings
    user_ratings = dict(zip(new_user_df['movie'], new_user_df['rating']))

    # Set up model parameters
    input_dim = len(movies_df['movie_id'].unique())  # Number of unique movies
    encoding_dim = 64
    model_path = "m_recommender_autoencoder_model.pth"

    # Load the trained model
    model = load_model(model_path, input_dim, encoding_dim)

    # Create user vector
    user_vector = create_user_vector(user_ratings, movie_to_id, input_dim)

    # Get top 10 recommendations
    recommendations = get_top_n_recommendations(model, user_vector, id_to_movie, n=10)

    # Print recommendations
    print("Top 10 movie recommendations:")
    for movie_title, predicted_rating in recommendations:
        print(f"{movie_title} - Predicted rating: {predicted_rating:.2f}")


if __name__ == "__main__":
    main()