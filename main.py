import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Text cleaning function"""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    return text.strip()

def preprocess_data(file_path):
    """Preprocessing function for data"""
  
    data = pd.read_csv(file_path)
    
    required_columns = ['article', 'highlights']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column {col} not found in the dataset")
    
    data['cleaned_article'] = data['article'].apply(clean_text)
    data['cleaned_highlights'] = data['highlights'].apply(clean_text)
    
    return data

def visualize_data(data):
    """Visualize the data"""
    plt.figure(figsize=(15, 10))
    
    # Plot the distribution of article lengths
    plt.subplot(2, 2, 1)
    data['article_length'] = data['cleaned_article'].apply(len)
    sns.histplot(data['article_length'], kde=True, color='blue')
    plt.title("Article Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    
    # Plot the distribution of summary lengths
    plt.subplot(2, 2, 2)
    data['summary_length'] = data['cleaned_highlights'].apply(len)
    sns.histplot(data['summary_length'], kde=True, color='red')
    plt.title("Summary Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    
    # Word cloud for articles
    plt.subplot(2, 2, 3)
    wordcloud_articles = WordCloud(width=400, height=200).generate(' '.join(data['cleaned_article']))
    plt.imshow(wordcloud_articles, interpolation='bilinear')
    plt.title("Word Cloud of Articles")
    plt.axis('off')
    
    # Word cloud for summaries
    plt.subplot(2, 2, 4)
    wordcloud_summaries = WordCloud(width=400, height=200).generate(' '.join(data['cleaned_highlights']))
    plt.imshow(wordcloud_summaries, interpolation='bilinear')
    plt.title("Word Cloud of Summaries")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def prepare_data_for_model(data):
    """Preparing the data for the model"""
    
    MAX_VOCAB_SIZE = 10000
    MAX_ARTICLE_LENGTH = 300
    MAX_SUMMARY_LENGTH = 100

    article_tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    summary_tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")

    article_tokenizer.fit_on_texts(data['cleaned_article'])
    summary_tokenizer.fit_on_texts(data['cleaned_highlights'])

    article_sequences = article_tokenizer.texts_to_sequences(data['cleaned_article'])
    summary_sequences = summary_tokenizer.texts_to_sequences(data['cleaned_highlights'])

    article_padded = pad_sequences(article_sequences, maxlen=MAX_ARTICLE_LENGTH, padding='post', truncating='post')
    summary_padded = pad_sequences(summary_sequences, maxlen=MAX_SUMMARY_LENGTH, padding='post', truncating='post')

    X_train, X_test, y_train, y_test = train_test_split(article_padded, summary_padded, test_size=0.2, random_state=42)

    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test,
        'article_tokenizer': article_tokenizer,
        'summary_tokenizer': summary_tokenizer
    }

def create_attention_model(vocab_size, max_input_length, max_output_length):
    """Creating the attention model for summarization"""
    
    encoder_inputs = Input(shape=(max_input_length,))
    decoder_inputs = Input(shape=(max_output_length,))
    
    embedding_dim = 256
    encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
    
    encoder_lstm = LSTM(512, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding(encoder_inputs))
    
    decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
    attention_layer = Attention()
    
    decoder_outputs_list = []
    decoder_state_h, decoder_state_c = state_h, state_c
    
    for t in range(max_output_length):
        decoder_x = decoder_inputs[:, t:t+1]
        decoder_embedding_x = decoder_embedding(decoder_x)
        
        context_vector = attention_layer([decoder_lstm_output, encoder_outputs])
        
        lstm_input = Concatenate()([decoder_embedding_x, context_vector])
        
        decoder_lstm_output, decoder_state_h, decoder_state_c = decoder_lstm(
            lstm_input, initial_state=[decoder_state_h, decoder_state_c]
        )
        
        decoder_outputs_list.append(decoder_lstm_output)
    
    decoder_outputs = tf.keras.layers.Concatenate(axis=1)(decoder_outputs_list)
    
    output_layer = Dense(vocab_size, activation='softmax')
    decoder_outputs = output_layer(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def train_model(data_dict):
    """Training the model"""
   
    model = create_attention_model(
        vocab_size=10000, 
        max_input_length=300, 
        max_output_length=100
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
   
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=0.00001
    )

    # Train the model
    history = model.fit(
        [data_dict['X_train'], data_dict['y_train'][:, :-1]], 
        data_dict['y_train'][:, 1:],
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr]
    )
    
    return model, history

def extract_and_save_model(model, tokenizers):
    """Extract and save the model and tokenizers"""
   
    os.makedirs('saved_models', exist_ok=True)
    
    model.save('saved_models/summarization_model.h5')
    
    import pickle
    
    with open('saved_models/article_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizers['article_tokenizer'], f)
    
    with open('saved_models/summary_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizers['summary_tokenizer'], f)
    
    print("Model and tokenizers saved successfully!")

def main():
    data_path = "cnn_dailymail_data/train.csv"
    
    preprocessed_data = preprocess_data(data_path)
    
    visualize_data(preprocessed_data)
    
    data_dict = prepare_data_for_model(preprocessed_data)
    
    model, training_history = train_model(data_dict)
    
    extract_and_save_model(model, data_dict)

if __name__ == "__main__":
    main()
