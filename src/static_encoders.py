import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoModelForSequenceClassification
from sklearn.preprocessing import normalize
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
# from nrclex import NRCLex
from tqdm.auto import tqdm
# import pysentiment2 as ps
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, models
from nltk.tokenize import sent_tokenize
import gc
import pandas as pd
import numpy as np
from collections import Counter
import torch
from tqdm.auto import tqdm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
# from transformers import pipeline, AutoTokenizer

def clean_and_validate(data):
    if isinstance(data, np.number):
        return str(data)
    elif isinstance(data, str):
        return data
    else:
        raise TypeError("Invalid type for text processing.")

def TFIDFEncoder(train, full, nums, top_n, n_grams=1):
    """
    Identifies the top n most overrepresented unigrams and bigrams in specified text columns 
    of the `train` DataFrame, applies TF-IDF vectorization to the corresponding columns 
    in the `full` DataFrame based on these terms, and saves the transformed `full` DataFrame to a file.

    Parameters:
    - train: DataFrame used to identify the top features in the training set.
    - full: DataFrame to apply the transformation.
    - nums: A list of numbers specifying which "mda_{num}" columns to process.
    - top_n: The number of features considered.
    """
    concatenated_texts_train = train[[f"mda_{num}" for num in nums]].apply(lambda x: ' '.join(x), axis=1)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, n_grams), max_features=top_n)
    vectorizer.fit(concatenated_texts_train)
    feature_names = vectorizer.get_feature_names_out()

    for num in nums:
        text_column_name = f"mda_{num}"
        tfidf_matrix_full = vectorizer.transform(full[text_column_name])
        tfidf_df_full = pd.DataFrame(tfidf_matrix_full.toarray(),
                                     columns=[f"{text_column_name}_tfidf_{feature}" for feature in feature_names])
        full = pd.concat([full, tfidf_df_full], axis=1)

    return full

def LDAEncoder(train, full, nums, n_topics=25, top_words=3):
    """
    Trains LDA models for multiple text columns identified by "mda_{num}" format in the `train` DataFrame,
    applies each topic model to the corresponding column in the `full` DataFrame,
    and saves the transformed `full` DataFrame to a file. Feature columns are named using the top words from each topic.

    Parameters:
    - train: DataFrame used to train the LDA models.
    - full: DataFrame to apply the LDA transformations.
    - nums: List of numbers identifying the text columns.
    - n_topics: Number of topics to identify per text column (default is 25).
    - top_words: Number of top words to use for naming each topic (default is 3).
    """
    concatenated_texts_train = train[[f"mda_{num}" for num in nums]].apply(lambda x: ' '.join(x.dropna()), axis=1)
    
    count_vect = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm_train = count_vect.fit_transform(concatenated_texts_train)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm_train)

    # Generate topic names
    feature_names = count_vect.get_feature_names_out()
    topic_names = ["_".join([feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]]) for topic in lda.components_]

    for num in nums:
        text_column_name = f"mda_{num}"
        dtm_full = count_vect.transform(full[text_column_name].dropna())
        lda_features_full = lda.transform(dtm_full)

        lda_features_full_normalized = normalize(lda_features_full, norm='l1', axis=1)
        lda_df_full = pd.DataFrame(lda_features_full_normalized,
                                   columns=[f"{text_column_name}_topic_{name}" for name in topic_names])
        full = pd.concat([full, lda_df_full], axis=1)

    return full


def NRCEncoder(df, nums, num_features=10):
    """
    Encodes specified text columns in a DataFrame using NRCLex to analyze emotional affect frequencies.
    
    Parameters:
    - df: DataFrame to process.
    - nums: List of numbers identifying the text columns.
    - num_features: Number of emotional affect features to include.
    
    Returns:
    - DataFrame with new columns for the emotional affect frequencies of each text entry.
    """
    
    for num in nums:
        text_column_name = f"mda_{num}"
        affect_frequencies_all_rows = []
        for text in df[text_column_name]:
            emotion = NRCLex(text)
            # Sort the affects by frequency and select the top num_features, if needed
            affect_freq_sorted = dict(sorted(emotion.affect_frequencies.items(), key=lambda item: item[1], reverse=True)[:num_features])
            affect_frequencies_all_rows.append(list(affect_freq_sorted.values()))
        
        # Use the sorted keys from the last item as column names, assuming uniformity across all texts
        affect_columns = [f"{text_column_name}_emotion_{affect}" for affect in affect_freq_sorted.keys()]
        affect_df = pd.DataFrame(affect_frequencies_all_rows, columns=affect_columns)
        df = pd.concat([df, affect_df], axis=1)

    return df
    

def LMEncoder(df, nums):
    """
    Encodes specified text columns in a DataFrame using pysentiment2's LM dictionary
    to analyze sentiment scores.
    
    Parameters:
    - df: DataFrame to process.
    - nums: Number of emotional affect features to include.
    
    Returns:
    - DataFrame with new normalized columns for the sentiment scores of each text entry.
    """
    lm = ps.LM()
    
    
    for num in nums:
        text_column_name = f"mda_{num}" 
        scores = {'Positive': [], 'Negative': [], 'Polarity': [], 'Subjectivity': []}
        
        for item in df[text_column_name]:
            tokens = lm.tokenize(item)
            score = lm.get_score(tokens)
            for key in scores:
                scores[key].append(score.get(key, 0))
        
        for key, values in scores.items():
            column_name = f"{text_column_name}_{key}"
            df[column_name] = values
    
    return df

def topic_dist(dict_, items):
    c = Counter(items)
    store = []
    for v in dict_.keys():
        if v == -1:
            continue
        elif v in items:
            store.append(c.get(v))
        else:
            store.append(0)

    if all(x == 0 for x in store):
        return [0] * len(store) 

    base = min(store)
    range = max(store) - base
    normalized = [(x-base) / (range if range > 0 else 1) for x in store]  # Avoid division by 0
    return normalized

def ClusterEncoder(train_df, full_df, nums, device, embedding_model='all-MiniLM-L6-v2', top_n_clusters=100):
    """
    Fits a clustering topic model on the `train_df` and applies the transformation to the `full_df`.
    
    Parameters:
    - train_df: DataFrame for training the model.
    - full_df: DataFrame to apply the transformations.
    - nums: Number of emotional affect features to include.
    - embedding_model: Model name for SentenceTransformer embeddings.
    - top_n_clusters: Number of topics/clusters.
    
    Returns:
    - Modified `full_df` with new columns for topic distributions for each text entry.
    """

    embedding_model = SentenceTransformer(embedding_model, device=device)
    bertopic_model = BERTopic(embedding_model=embedding_model, nr_topics=top_n_clusters, verbose=True)

    all_train_sentences = []
    for num in nums:
        text_column_name = f"mda_{num}"
        for text in tqdm(train_df[text_column_name].dropna(), desc=f"Training on {text_column_name}"):
            all_train_sentences.extend(sent_tokenize(text))

    all_train_sentences = all_train_sentences[:10000]
    train_embeddings = embedding_model.encode(all_train_sentences, show_progress_bar=True)
    bertopic_model.fit(all_train_sentences, train_embeddings)
    topic_info = bertopic_model.get_topic_info()
    topic_name_mapping = topic_info.set_index('Topic')['Name'].to_dict()
    topic_name_mapping.pop(next(iter(topic_name_mapping)))

    print("Model created!")
    
    for num in nums:
        text_column_name = f"mda_{num}"
        topic_distributions = []
        for text in tqdm(full_df[text_column_name].dropna(), desc=f"Transforming {text_column_name}"):
            text = clean_and_validate(text)
            sentences = sent_tokenize(str(text))  
            sentences = [str(sentence) for sentence in sentences if sentence]  

            for sentence in sentences:
                if not isinstance(sentence, str):
                    raise ValueError(f"Expected a string, but got {type(sentence)}: {sentence}")
                assert isinstance(sentence, str), f"Non-string sentence found: {sentence} with type {type(sentence)}"
                assert len(sentence) > 0, "Empty string found as sentence"
            
            if sentences:
                if isinstance(sentences, np.ndarray):
                    sentences = sentences.tolist()

                if not all(isinstance(s, str) for s in sentences):
                    raise ValueError("All elements in sentences must be strings.")

                topics, probs = bertopic_model.transform(sentences)

                norm_distribution = topic_dist(topic_name_mapping, topics)

                topic_distributions.append(norm_distribution)
            else:
                topic_distributions.append(np.zeros(len(topic_name_mapping)))
            gc.collect()
            del topics
            del probs

        for i, topic_num in enumerate(sorted(topic_name_mapping.keys())):
            topic_name = topic_name_mapping[topic_num]
            full_df[f"{text_column_name}_topic_{topic_name}"] = [dist[i] for dist in topic_distributions]

    return full_df, bertopic_model


def ClusterEncoder_(train_df, full_df, nums, device, embedding_model='all-MiniLM-L6-v2', top_n_clusters=100):
    """
    Fits a clustering topic model on the `train_df` and applies the transformation to the `full_df`.
    
    Args:
    - train_df (pd.DataFrame): DataFrame for training the model.
    - full_df (pd.DataFrame): DataFrame to apply the transformations.
    - nums (list[int]): Number of emotional affect features to include.
    - device (str): The device to run the model on ('cpu' or 'cuda').
    - embedding_model (str): Model name for SentenceTransformer embeddings.
    - top_n_clusters (int): Number of topics/clusters.
    
    Returns:
    - pd.DataFrame: Modified `full_df` with new columns for topic distributions for each text entry.
    - BERTopic: Trained BERTopic model.
    """
    print("spag_bowl")
    sentence_model = SentenceTransformer(embedding_model, device=device)
    
    bertopic_model = BERTopic(embedding_model=sentence_model, nr_topics=top_n_clusters, calculate_probabilities=True, verbose=True)

    all_train_sentences = []
    for num in nums:
        text_column_name = f"mda_{num}"
        for text in tqdm(train_df[text_column_name].dropna(), desc=f"Training on {text_column_name}"):
            all_train_sentences.extend(sent_tokenize(text))

    print(all_train_sentences[:2])

    # train_embeddings = embedding_model.encode(all_train_sentences, show_progress_bar=True)
    all_train_sentences = all_train_sentences[:2000]
    
    bertopic_model.fit(all_train_sentences)
    topic_info = bertopic_model.get_topic_info()
    topic_name_mapping = topic_info.set_index('Topic')['Name'].to_dict()
    print(topic_info)
    print(topic_name_mapping)

    
    for num in nums:
        column_name = f"mda_{num}"
        topic_distribution_list = []

        for text in tqdm(full_df[column_name].fillna(""), desc=f"Encoding {column_name}"):
            if text:
                topic_distr, topic_token_distr = bertopic_model.approximate_distribution(text, calculate_tokens=True)
                print(f"td: {topic_distr}")  # Print full "td" output for diagnosis
                print(f"td length: {len(topic_distr)}")  # Print distribution length
                topic_distribution_list.append(topic_distr)
            else:
                # Handle empty text (e.g., fill with zeros)
                max_length = max(len(dist) for dist in topic_distribution_list or [[]])
                topic_distribution_list.append(np.zeros(max_length))

        # No normalization needed (commented out)

        for i, topic_num in enumerate(sorted(topic_name_mapping.keys())):
            if topic_num == -1:
                continue
            topic_name = topic_name_mapping[topic_num]
            try:
                full_df[f"{column_name}_topic_{topic_name}"] = [dist[i] for dist in topic_distribution_list]
            except IndexError:
                # Handle missing distribution (e.g., fill with zeros)
                full_df[f"{column_name}_topic_{topic_name}"] = [0 for _ in range(len(topic_distribution_list[0]))]

        print(full_df.head())
        print(full_df.shape)

        return full_df, bertopic_model

def BERTEmotionEncoder(full_df, nums, model_name="j-hartmann/emotion-english-distilroberta-base", num_features=7):
    """
    Applies an emotion text classification model to encode texts from specified columns in `full_df`.
    
    Parameters:
    - full_df: DataFrame to apply the emotion text classifier transformations.
    - nums: Number of emotional affect features to include.
    - model_name: The identifier for a pre-trained emotion text classification model.
    - num_features: The number of emotion categories the model classifies.
    
    Returns:
    - `full_df` modified with new columns for emotion classification scores for each text entry.
    """
    device = 0 if torch.cuda.is_available() else -1  # Assuming device is meant for setting model device
    classifier = pipeline("text-classification", model=model_name, return_all_scores=True, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define a mapping from emotion labels to indices based on the classifier's output
    emotion_labels = classifier.model.config.id2label.values()  # Get emotion labels from the model config
    emotion_to_index = {label: i for i, label in enumerate(emotion_labels)}

    def process_text_in_chunks(text):
        # Tokenize the text and split into chunks by words to avoid splitting tokens
        words = text.split()
        max_length = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add()
        chunk_texts = [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
        
        all_scores = np.zeros((len(chunk_texts), len(emotion_labels)))
        for chunk_text in chunk_texts:
            outputs = classifier(chunk_text, truncation=True, padding=True)
            chunk_scores = np.zeros(len(emotion_labels))
            for output in outputs[0]:  # Access the predictions for the chunk
                emotion = output['label']
                if emotion in emotion_to_index:
                    index = emotion_to_index[emotion]
                    chunk_scores[index] = output['score']
            all_scores = np.vstack([all_scores, chunk_scores])  # Stack the scores
        
        # Calculate mean scores across all chunks
        mean_scores = np.mean(all_scores, axis=0)
        return mean_scores

    for num in nums:
        text_column_name = f"mda_{num}"
        print(f"Processing column: {text_column_name}")
        
        # Initialize dictionaries to hold scores for each emotion
        emotion_scores = {label: [] for label in emotion_labels}
        
        # Process each text entry
        for text in tqdm(full_df[text_column_name].fillna(""), desc=f"Encoding {text_column_name}"):
            scores = process_text_in_chunks(text)
            for label, score in zip(emotion_labels, scores):
                emotion_scores[label].append(score)
        
        # Update full_df with new columns for each emotion score
        for label, scores in emotion_scores.items():
            column_name = f"{text_column_name}_emotion_{label}"
            full_df[column_name] = scores

    return full_df

def BERTEncoder(full_df, nums, model_name='bert-base-uncased', num_features=768):
    """
    Applies a BERT model to encode texts from specified columns in `full_df`.
    
    Parameters:
    - full_df: DataFrame to apply the BERT transformations.
    - nums: Number of emotional affect features to include.
    - model_name: The identifier for a pre-trained BERT model.
    - num_features: The number of features to extract from the BERT embeddings.
    
    Returns:
    - `full_df` modified with new columns for BERT embeddings for each text entry.
    """
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    for num in nums:
        text_column_name = f"mda_{num}" 
        embeddings_matrix = np.zeros((len(full_df), num_features))
        
        for i, text in enumerate(tqdm(full_df[text_column_name].fillna(""), desc=f"Encoding {text_column_name}")):
            embedding = bert_model.encode(text, convert_to_numpy=True)
            embeddings_matrix[i, :len(embedding)] = embedding[:num_features]
        
        for i in range(num_features):
            full_df[f"{text_column_name}_bert_feature_{i}"] = embeddings_matrix[:, i]

    return full_df


def LongformerEncoder(full_df, nums, device, model_name='allenai/longformer-base-4096', num_features=768):
    """
    Applies a Longformer model to encode texts from specified columns in `full_df` using CUDA if available.
    
    Parameters:
    - full_df: DataFrame to apply the Longformer transformations.
    - nums: Number of emotional affect features to include.
    - model_name: The identifier for a pre-trained Longformer model.
    - num_features: The number of features to extract from the Longformer embeddings.
    
    Returns:
    - `full_df` modified with new columns for Longformer embeddings for each text entry.
    """
    word_embedding_model = models.Transformer(model_name, max_seq_length=4096)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    
    longformer_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    longformer_model.to(device)  

    for num in nums:
        text_column_name = f"mda_{num}"
        embeddings_matrix = np.zeros((len(full_df), num_features))
        
        for i, text in enumerate(tqdm(full_df[text_column_name].fillna(""), desc=f"Encoding {text_column_name}")):
            embedding = longformer_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
            embedding = embedding.to('cpu').numpy()  
            embeddings_matrix[i, :len(embedding)] = embedding[:num_features]
        
        for i in range(num_features):
            full_df[f"{text_column_name}_longformer_feature_{i}"] = embeddings_matrix[:, i]

    return full_df




################################## chi-squared ######################################


def train_and_transform_bertopic_with_feature_selection(train_df, full_df, text_columns, target_column='change', n_features=100, embedding_model='all-MiniLM-L6-v2', top_n_clusters=100):
    """
    Fits a BERTopic model and selects features using a chi-squared test on the `train_df`, then
    applies the BERTopic transformation and feature selection to the `full_df`.
    
    Parameters:
    - train_df: DataFrame for training the BERTopic model and feature selection.
    - full_df: DataFrame to apply the BERTopic transformations and selected features.
    - text_columns: List of text column names to process.
    - target_column: The name of the target variable column.
    - n_features: Number of top features to select with chi-squared test.
    - embedding_model: Model name for SentenceTransformer embeddings.
    - top_n_clusters: Number of topics/clusters for BERTopic.
    
    Returns:
    - Modified `full_df` with new columns for selected topic distributions.
    - The trained BERTopic model.
    - The SelectKBest model for feature selection.
    """
    # Initialize the embedding model and BERTopic model
    embedding_model = SentenceTransformer(embedding_model)
    bertopic_model = BERTopic(nr_topics=top_n_clusters, verbose=True)
    
    # Apply BERTopic transformation to train_df and encode features
    train_encoded, bertopic_model = apply_bertopic_transform(train_df, text_columns, embedding_model, bertopic_model)
    
    # Select the top n_features using chi-squared test
    selected_features, chi2_selector = select_features_chi_squared(train_encoded, target_column, n_features)
    
    # Apply BERTopic transformation to full_df
    full_encoded, _ = apply_bertopic_transform(full_df, text_columns, embedding_model, bertopic_model, selected_features.columns)
    
    return full_encoded, bertopic_model, chi2_selector

def select_features_chi_squared(df, target_column, n_features):
    """
    Selects the top n_features most predictive of the target variable using a chi-squared test.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    chi2_selector = SelectKBest(chi2, k=n_features)
    X_kbest = chi2_selector.fit_transform(X_scaled, y)
    selected_features = df.columns[chi2_selector.get_support(indices=True)]
    return df[selected_features], chi2_selector

def apply_bertopic_transform(df, text_columns, embedding_model, bertopic_model, fit_model=False):
    """
    Applies BERTopic transformation to a DataFrame for specified text columns, 
    generating topic distributions for each document.
    
    Parameters:
    - df: DataFrame to process.
    - text_columns: List of text column names to process with BERTopic.
    - embedding_model: Initialized SentenceTransformer model for embeddings.
    - bertopic_model: Initialized or previously fitted BERTopic model.
    - fit_model: Boolean indicating if the BERTopic model should be fit to this data.
    
    Returns:
    - df_encoded: DataFrame with new columns for topic distributions.
    - bertopic_model: The (fitted) BERTopic model.
    """
    all_sentences = []
    sentence_to_text_mapping = []
    
    # Tokenize and prepare sentences from all specified text columns
    for col in text_columns:
        for idx, text in enumerate(df[col].fillna("")):
            sentences = sent_tokenize(text)
            all_sentences.extend(sentences)
            sentence_to_text_mapping.extend([(idx, col) for _ in sentences])
    
    # Generate embeddings for all sentences
    if all_sentences:
        sentence_embeddings = embedding_model.encode(all_sentences, show_progress_bar=True)
    
        # Fit or transform with BERTopic
        if fit_model:
            topics, _ = bertopic_model.fit_transform(all_sentences, embeddings=sentence_embeddings)
        else:
            topics, _ = bertopic_model.transform(all_sentences)
    
        # Prepare topic distribution per document
        df_encoded = pd.DataFrame(index=df.index, columns=[f"topic_{i}" for i in range(bertopic_model.nr_topics)])
        df_encoded.fillna(0, inplace=True)  # Initialize with zeros
        
        for (doc_id, col), topic in zip(sentence_to_text_mapping, topics):
            df_encoded.loc[doc_id, f"topic_{topic}"] += 1
    
        # Normalize topic counts to distributions
        df_encoded = df_encoded.div(df_encoded.sum(axis=1), axis=0).fillna(0)
        
        # Concatenate original df columns with topic distribution columns
        df_encoded = pd.concat([df, df_encoded], axis=1)
    else:
        # If no sentences are found, create an empty DataFrame with topic distribution columns
        df_encoded = pd.DataFrame(columns=[f"topic_{i}" for i in range(bertopic_model.nr_topics)])
    
    return df_encoded, bertopic_model