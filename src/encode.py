from static_encoders import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default="lm_lex", help='Type of Text Encoder')
parser.add_argument('--lag', type=int, default=0, help='Number of lags')
parser.add_argument('--n_grams', type=int, default=1, help='Number of n_grams included in tf-idf')
parser.add_argument('--sentence_model', type=str, default="all-roberta-large-v1", help='Sentence model for clusters topic model.')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

threshold_dict = {0: '2005Q3', 
                  1: '2009Q1', 
                  2: '2009Q3', 
                  3: '2010Q2'}

def load_datasets(lag, threshold_dict):
    """
    Load train and full datasets for a given lag. Train represents the first 1000 samples for each lag. 

    Parameters:
    - lag (int): The specific lag for which to load the datasets. Assumes file path naming convention based on lag.

    Returns:
    - tuple of pandas.DataFrame: train and full dataframe for each lag
    """

    full = pd.read_csv(f"../Data/final_dataframes/lag{lag}/text/full.csv")

    training_cutoff = threshold_dict[lag]
    full['quarter_dt'] = pd.PeriodIndex(full['quarter'], freq='Q').to_timestamp()
    training_cutoff_dt = pd.Period(training_cutoff, freq='Q').to_timestamp()

    validation_cutoff_dt = training_cutoff_dt + pd.DateOffset(years=1)
    test_cutoff_dt = validation_cutoff_dt + pd.DateOffset(months=3)

    train = full[full['quarter_dt'] <= training_cutoff_dt]
    
    return train, full  
    

def save_df(df, encoder_name, lag, n_grams, sentence_model):

    directory_path = f"dataframes/lag{lag}"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if encoder_name == "tfidf":
        if n_grams == 1:
            save_path = f"{directory_path}/{encoder_name}_unigrams.csv"
        elif n_grams == 2:
            save_path = f"{directory_path}/{encoder_name}_bigrams.csv"
        elif n_grams == 3:
            save_path = f"{directory_path}/{encoder_name}_trigrams.csv"
        else:
            ValueError("n_grams has an invalid value out of the scope of our testing. Valid values are 1, 2 or 3.")
    elif encoder_name == "clusters":
        save_path = f"{directory_path}/{encoder_name}_{sentence_model}.csv"
    else:
        save_path = f"{directory_path}/{encoder_name}.csv"

    df.to_csv(save_path)

if __name__ == "__main__":
    lag = args.lag
    for lag in [0, 1, 2, 3]:
        print("Lag is ", lag)
        nums = [i for i in range(lag+1)]
        print("Columns considered", nums)
        train, full = load_datasets(lag, threshold_dict)

        if args.encoder == "lda":
            df = LDAEncoder(train, full, nums, n_topics=25)
        elif args.encoder == "tfidf":
            df = TFIDFEncoder(train, full, nums, top_n=50, n_grams=args.n_grams)
        elif args.encoder == "clusters":
            df, bertopic_model = ClusterEncoder(train, full, nums, device, embedding_model=args.sentence_model)
            bertopic_model.save(f"bertopic_model_{lag}", serialization="pickle")
        elif args.encoder == "lm_lex":
            df = LMEncoder(full, nums)
        elif args.encoder == "nrc_lex":
            df = NRCEncoder(full, nums)
        elif args.encoder == "bert":
            df = BERTEncoder(full, nums)
        elif args.encoder == "longformer":
            df = LongformerEncoder(full, nums, device)
        elif args.encoder == "bert_emo":
            df = BERTEmotionEncoder(full, nums)
        
        print("Dataframe done!")

        cutoff_str = threshold_dict[lag]
        cutoff_date = pd.Period(cutoff_str, freq='Q').to_timestamp()
        new_columns = [col for col in df.columns if col not in full.columns]
        filtered_df = df[df['quarter_dt'] <= cutoff_date]

        for col in new_columns:
            max_value = filtered_df[col].max()
            if max_value != 0:
                df[col] = df[col] / max_value

        save_df(df, args.encoder, lag, args.n_grams, args.sentence_model)
        
        print("Saved!!")