import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import scipy.sparse as sp

# Check if CUDA is available
device = "cuda"

# Load the pre-processed CSV file
print("Loading pre-processed ratings dataset...")
df = pd.read_csv('netflix_ratings.csv')

# Ensure correct data types
df['movie_id'] = df['movie_id'].astype('int32')
df['user_id'] = df['user_id'].astype('int32')
df['rating'] = df['rating'].astype('float32')

# Create sparse matrix
print("Creating sparse user-movie matrix...")
user_ids = df['user_id'].astype('category').cat.codes
movie_ids = df['movie_id'].astype('category').cat.codes
ratings = df['rating'].values

shape = (470758, 4499)  # Use the provided dimensions
user_movie_matrix = csr_matrix((ratings, (user_ids, movie_ids)), shape=shape)

#print("Saving sparse matrix as NPZ...")
#sp.save_npz('user_movie_matrix.npz', user_movie_matrix)

print("\nDataset Information:")
print(f"Number of users (samples): {user_movie_matrix.shape[0]}")
print(f"Number of movies (features): {user_movie_matrix.shape[1]}")
print(f"Total number of ratings: {user_movie_matrix.nnz}")
print(f"Sparsity: {user_movie_matrix.nnz / (user_movie_matrix.shape[0] * user_movie_matrix.shape[1]):.4f}")

# Split the data
print("\nSplitting data...")
X_train, X_test = train_test_split(user_movie_matrix, test_size=0.2, random_state=1)

# Convert to PyTorch sparse tensors
def sparse_matrix_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

X_train_tensor = sparse_matrix_to_torch_sparse_tensor(X_train).to(device)
X_test_tensor = sparse_matrix_to_torch_sparse_tensor(X_test).to(device)

# Custom Dataset for sparse tensors
class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_tensor):
        self.sparse_tensor = sparse_tensor

    def __len__(self):
        return self.sparse_tensor.shape[0]

    def __getitem__(self, idx):
        return self.sparse_tensor[idx]

def sparse_collate(batch):
    return torch.stack(batch)

# Create DataLoaders
batch_size = 128  # Adjusted batch size
train_dataset = SparseDataset(X_train_tensor)
test_dataset = SparseDataset(X_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sparse_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=sparse_collate)

# Define the autoencoder architecture with dropout
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

    def encode(self, x):
        if x.is_sparse:
            x = x.to_dense()
        return self.encoder(x)

# Initialize the model
input_dim = X_train.shape[1]
encoding_dim = 64
model = Autoencoder(input_dim, encoding_dim, dropout_rate=0.1).to(device)

# Define custom loss function (unchanged)
class SparseRatingLoss(nn.Module):
    def __init__(self):
        super(SparseRatingLoss, self).__init__()

    def forward(self, pred, true):
        mask = true != 0
        loss = F.mse_loss(pred[mask], true[mask], reduction='mean')
        return loss

criterion = SparseRatingLoss()

# Add L2 regularization by using weight decay in the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Added weight decay for L2 regularization

# The rest of the code remains the same
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

# Function to evaluate the model
def evaluate_model(model, data_loader, criterion, pbar=None):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in pbar if pbar is not None else data_loader:
            inputs = batch.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, inputs.to_dense())

            total_loss += loss.item() * inputs._nnz()
            total_samples += inputs._nnz()

            all_true.extend(inputs._values().cpu().numpy())
            all_pred.extend(outputs[inputs._indices()[0], inputs._indices()[1]].cpu().numpy())

            if pbar is not None:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_samples

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    mse = mean_squared_error(all_true, all_pred)
    rmse = np.sqrt(mse)

    return avg_loss, rmse

# Training the model
epochs = 30
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    total_samples = 0

    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
    for batch in train_pbar:
        inputs = batch.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, inputs.to_dense())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs._nnz()
        total_samples += inputs._nnz()

        train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'rmse': f'{(loss.item() ** 0.5):.4f}'})

    epoch_loss /= total_samples

    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}, RMSE: {epoch_loss ** 0.5:.4f}')

    val_loss, val_rmse = evaluate_model(model, test_loader, criterion, tqdm(test_loader, desc='Validating', leave=False))
    print(f'Validation Loss: {val_loss:.4f}, Validation RMSE: {val_rmse:.4f}')

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    scheduler.step(val_loss)

# Load the best model weights
model.load_state_dict(best_model_wts)

print("Training completed.")

# Save the model
print("Saving the model...")
torch.save(model.state_dict(), 'm_recommender_autoencoder_model.pth')
print("Model saved successfully.")