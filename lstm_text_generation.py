"""
Text Generation System using LSTM
===================================
This module implements a character-level text generation model using stacked LSTM layers
trained on the Tiny Shakespeare dataset.

Temperature Concept:
-------------------
Temperature is a parameter that controls the randomness of predictions during text generation.
- Low temperature (e.g., 0.2): Makes the model more confident and conservative, choosing 
  high-probability characters. Results in more predictable, repetitive text.
- Medium temperature (e.g., 0.5-0.7): Balanced approach between creativity and coherence.
- High temperature (e.g., 1.0-1.5): Increases randomness, allowing the model to take risks 
  and choose less probable characters. Results in more creative but potentially nonsensical text.

Mathematically, temperature scales the logits before applying softmax:
P(next_char) = softmax(logits / temperature)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import urllib.request
import os
import string


# ============================================================================
# DATA DOWNLOAD AND PREPROCESSING
# ============================================================================

def download_shakespeare_data(filepath='shakespeare.txt'):
    """
    Download the Tiny Shakespeare dataset.
    
    Args:
        filepath: Path to save the downloaded file
        
    Returns:
        Path to the downloaded file
    """
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    
    if not os.path.exists(filepath):
        print(f"Downloading Shakespeare dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Dataset downloaded to {filepath}")
    else:
        print(f"Dataset already exists at {filepath}")
    
    return filepath


def preprocess_text(text):
    """
    Preprocess the text: convert to lowercase, remove punctuation and non-ASCII characters.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def create_character_mappings(text):
    """
    Create character-to-integer and integer-to-character mappings.
    
    Args:
        text: Preprocessed text
        
    Returns:
        char_to_int: Dictionary mapping characters to integers
        int_to_char: Dictionary mapping integers to characters
        vocab_size: Number of unique characters
    """
    # Get unique characters
    unique_chars = sorted(list(set(text)))
    
    # Create mappings
    char_to_int = {char: idx for idx, char in enumerate(unique_chars)}
    int_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    
    vocab_size = len(unique_chars)
    
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Sample characters: {unique_chars[:20]}")
    
    return char_to_int, int_to_char, vocab_size


def create_sliding_window_dataset(text, char_to_int, seq_length=100):
    """
    Create sliding window dataset with sequence length of 100 and next character as target.
    
    Args:
        text: Preprocessed text
        char_to_int: Character to integer mapping
        seq_length: Length of input sequences (default: 100)
        
    Returns:
        X: Input sequences (encoded)
        y: Target characters (encoded)
    """
    # Encode the entire text
    encoded_text = [char_to_int[char] for char in text]
    
    X = []
    y = []
    
    # Create sliding windows
    for i in range(len(encoded_text) - seq_length):
        # Input: seq_length characters
        seq_in = encoded_text[i:i + seq_length]
        # Target: next character
        seq_out = encoded_text[i + seq_length]
        
        X.append(seq_in)
        y.append(seq_out)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset created: {X.shape[0]} sequences")
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    return X, y


def prepare_data_for_training(X, y, vocab_size, test_split=0.1):
    """
    Prepare data for training: one-hot encode targets and split into train/val.
    
    Args:
        X: Input sequences
        y: Target characters
        vocab_size: Vocabulary size
        test_split: Fraction of data to use for validation
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    # One-hot encode the output
    y_one_hot = keras.utils.to_categorical(y, num_classes=vocab_size)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y_one_hot = y_one_hot[indices]
    
    # Split into training and validation sets
    split_idx = int(len(X) * (1 - test_split))
    
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    y_train = y_one_hot[:split_idx]
    y_val = y_one_hot[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_lstm_model(seq_length, vocab_size, embedding_dim=128, lstm_units=[256, 256], 
                     dropout_rate=0.2, learning_rate=0.001):
    """
    Build LSTM model with embedding layer, stacked LSTM layers, and dense output layer.
    
    Architecture:
    - Embedding Layer: Learn character representations
    - LSTM Layer 1: Process sequences with return_sequences=True
    - Dropout: Prevent overfitting
    - LSTM Layer 2: Further process temporal patterns
    - Dropout: Prevent overfitting
    - Dense Layer: Output probabilities for each character
    
    Args:
        seq_length: Length of input sequences
        vocab_size: Number of unique characters
        embedding_dim: Dimension of embedding vectors
        lstm_units: List of units for each LSTM layer
        dropout_rate: Dropout rate
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Embedding Layer: Convert character indices to dense vectors
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                        input_length=seq_length))
    
    # First LSTM Layer with return_sequences=True to feed into next LSTM
    model.add(LSTM(lstm_units[0], return_sequences=True, 
                   input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM Layer
    model.add(LSTM(lstm_units[1]))
    model.add(Dropout(dropout_rate))
    
    # Dense Output Layer with softmax activation
    model.add(Dense(vocab_size, activation='softmax'))
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("Model Architecture:")
    model.summary()
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=64, 
                checkpoint_path='best_model.h5'):
    """
    Train the LSTM model with callbacks for checkpointing and early stopping.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size for training
        checkpoint_path: Path to save best model
        
    Returns:
        Training history
    """
    # Define callbacks
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    
    print(f"\nStarting training for {epochs} epochs with batch size {batch_size}...")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    print(f"Training completed! Best model saved to {checkpoint_path}")
    
    return history


# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_text(model, seed_text, char_to_int, int_to_char, seq_length=100, 
                  next_chars=200, temperature=0.7):
    """
    Generate text character by character using the trained LSTM model.
    
    Temperature controls the randomness of predictions:
    - Low temperature (< 0.5): More conservative, predictable text
    - Medium temperature (0.5-0.8): Balanced creativity and coherence
    - High temperature (> 0.8): More creative, risky predictions
    
    The temperature is applied by dividing the logits before softmax,
    which flattens (high temp) or sharpens (low temp) the probability distribution.
    
    Args:
        model: Trained Keras model
        seed_text: Starting text string
        char_to_int: Character to integer mapping
        int_to_char: Integer to character mapping
        seq_length: Expected input sequence length
        next_chars: Number of characters to generate
        temperature: Controls randomness (higher = more random)
        
    Returns:
        Generated text string
    """
    # Start with the seed text
    current_text = seed_text.lower()
    generated_text = current_text
    
    # Ensure seed text is at least seq_length characters
    if len(current_text) < seq_length:
        # Pad with spaces if needed
        current_text = ' ' * (seq_length - len(current_text)) + current_text
    
    print(f"\nGenerating {next_chars} characters with temperature={temperature}...")
    print(f"Seed text: '{seed_text}'")
    print("-" * 80)
    
    for i in range(next_chars):
        # Extract the last seq_length characters
        input_seq = current_text[-seq_length:]
        
        # Encode the input sequence
        encoded_input = [char_to_int[char] for char in input_seq]
        encoded_input = np.array(encoded_input).reshape(1, -1)
        
        # Get prediction probabilities
        pred_probs = model.predict(encoded_input, verbose=0)[0]
        
        # Apply temperature scaling
        # Temperature adjusts the probability distribution:
        # - High temp: flattens distribution (more uniform, more random)
        # - Low temp: sharpens distribution (more peaked, less random)
        pred_logits = np.log(pred_probs + 1e-8)  # Add small value to avoid log(0)
        pred_logits_adjusted = pred_logits / temperature
        
        # Convert back to probabilities with temperature adjustment
        pred_probs_adjusted = np.exp(pred_logits_adjusted)
        pred_probs_adjusted = pred_probs_adjusted / np.sum(pred_probs_adjusted)
        
        # Sample from the adjusted probability distribution
        next_char_idx = np.random.choice(len(pred_probs_adjusted), 
                                          p=pred_probs_adjusted)
        next_char = int_to_char[next_char_idx]
        
        # Append to generated text
        generated_text += next_char
        
        # Update current text for next iteration
        current_text += next_char
        
        # Print progress
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{next_chars} characters...")
    
    print("-" * 80)
    return generated_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to orchestrate the complete text generation pipeline.
    """
    print("=" * 80)
    print("TEXT GENERATION SYSTEM USING LSTM")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Download and Preprocess Data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: DATA DOWNLOAD AND PREPROCESSING")
    print("=" * 80)
    
    # Download dataset
    data_path = download_shakespeare_data()
    
    # Load text
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\nOriginal text length: {len(text)} characters")
    print(f"First 200 characters:\n{text[:200]}")
    
    # Preprocess text
    print("\nPreprocessing text...")
    text = preprocess_text(text)
    print(f"Cleaned text length: {len(text)} characters")
    print(f"First 200 characters:\n{text[:200]}")
    
    # Create character mappings
    print("\nCreating character mappings...")
    char_to_int, int_to_char, vocab_size = create_character_mappings(text)
    
    # Create sliding window dataset
    print("\nCreating sliding window dataset...")
    seq_length = 100
    X, y = create_sliding_window_dataset(text, char_to_int, seq_length)
    
    # Prepare data for training
    print("\nPreparing data for training...")
    X_train, X_val, y_train, y_val = prepare_data_for_training(X, y, vocab_size)
    
    # -------------------------------------------------------------------------
    # Step 2: Build Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING LSTM MODEL")
    print("=" * 80)
    
    model = build_lstm_model(
        seq_length=seq_length,
        vocab_size=vocab_size,
        embedding_dim=128,
        lstm_units=[256, 256],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Train Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING MODEL")
    print("=" * 80)
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=15,
        batch_size=64,
        checkpoint_path='lstm_shakespeare_best.h5'
    )
    
    # Save the final model as well
    model.save('lstm_shakespeare_final.h5')
    print("\nFinal model saved to lstm_shakespeare_final.h5")
    
    # -------------------------------------------------------------------------
    # Step 4: Text Generation Demonstration
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: TEXT GENERATION DEMONSTRATION")
    print("=" * 80)
    
    # Load the best model
    print("Loading best model...")
    best_model = keras.models.load_model('lstm_shakespeare_best.h5')
    
    # Demonstrate generation with three different seeds
    seed_texts = [
        "to be or not to be",
        "what light through yonder window breaks",
        "all the world's a stage"
    ]
    
    temperatures = [0.5, 0.7, 1.0]
    
    for seed in seed_texts:
        print(f"\n{'=' * 80}")
        print(f"SEED: '{seed}'")
        print(f"{'=' * 80}")
        
        for temp in temperatures:
            generated = generate_text(
                model=best_model,
                seed_text=seed,
                char_to_int=char_to_int,
                int_to_char=int_to_char,
                seq_length=seq_length,
                next_chars=200,
                temperature=temp
            )
            print(f"\nTemperature {temp}:")
            print(generated)
            print()
    
    # Additional demonstration with custom parameters
    print("\n" + "=" * 80)
    print("ADDITIONAL GENERATION EXAMPLES")
    print("=" * 80)
    
    custom_seeds = [
        "romeo romeo wherefore art thou",
        "friends romans countrymen lend me your ears",
        "once upon a time"
    ]
    
    for seed in custom_seeds:
        generated = generate_text(
            model=best_model,
            seed_text=seed,
            char_to_int=char_to_int,
            int_to_char=int_to_char,
            seq_length=seq_length,
            next_chars=150,
            temperature=0.7
        )
        print(f"\nSeed: '{seed}'")
        print(f"Generated text:\n{generated[len(seed):]}")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("TEXT GENERATION SYSTEM COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
