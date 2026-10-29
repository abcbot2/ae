import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

vocab_size, max_len = 10000, 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=max_len, padding="post").astype("float32") / vocab_size
x_test = pad_sequences(x_test, maxlen=max_len, padding="post").astype("float32") / vocab_size

inputs = Input(shape=(max_len,))
x = Dense(128, activation="relu")(inputs)
x = Dense(64, activation="relu")(x)
latent = Dense(32, activation="relu")(x)

x = Dense(64, activation="relu")(latent)
x = Dense(128, activation="relu")(x)
outputs = Dense(max_len, activation="sigmoid")(x)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(x_train, x_train, epochs=5, batch_size=128, validation_split=0.2)

encoder = Model(inputs, latent)
train_features = encoder.predict(x_train, verbose=0)
test_features = encoder.predict(x_test, verbose=0)

classifier_input = Input(shape=(32,))
x = Dense(16, activation="relu")(classifier_input)
output = Dense(1, activation="sigmoid")(x)

classifier = Model(classifier_input, output)
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
classifier.fit(train_features, y_train, epochs=5, batch_size=128)

predictions = (classifier.predict(test_features, verbose=0) > 0.5).astype(int).ravel()
print("Test accuracy:", accuracy_score(y_test, predictions))

-----------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda image: image.view(-1))
])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 256)
        self.mean = nn.Linear(256, 2)
        self.log_var = nn.Linear(256, 2)
        self.decoder1 = nn.Linear(2, 256)
        self.decoder2 = nn.Linear(256, 784)

    def forward(self, x):
        h = F.relu(self.encoder(x))
        mean = self.mean(h)
        log_var = self.log_var(h)

        std = torch.exp(0.5 * log_var)
        z = mean + std * torch.randn_like(std)

        reconstruction = torch.sigmoid(self.decoder2(F.relu(self.decoder1(z))))
        return reconstruction, mean, log_var

def vae_loss(reconstruction, x, mean, log_var):
    reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_loss

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0

    for images, _ in train_loader:
        optimizer.zero_grad()
        reconstruction, mean, log_var = model(images)
        loss = vae_loss(reconstruction, images, mean, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, loss: {total_loss / len(train_data):.4f}")
