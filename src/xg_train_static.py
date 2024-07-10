import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import matplotlib.pyplot as plt
import os
import argparse
import xgboost as xgb
from tqdm import tqdm
from tqdm.auto import tqdm
import json
import nltk
import joblib
nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument('--lag', type=int, default=0, help='Quarters of lag.')
parser.add_argument('--feature_type', type=str, default='nrc_lex', help='Method for encoding text.')
parser.add_argument('--no_text', type=bool, default=False, help='ONLY FOR RF MODEL - no text included in calculation.')
parser.add_argument('--no_text_features', type=str, default='all', help='ONLY FOR RF MODEL - no text included in calculation.')
parser.add_argument('--only_text', type=bool, default=False, help='ONLY FOR RF MODEL - no text included in calculation.')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
    
def extract_text_before_last_underscore(s):
    last_underscore_index = s.rfind('_')
    return s[:last_underscore_index]

class ModelTrainer:
    def __init__(self, args):
        self.feature_type = args.feature_type
        self.threshold_dict = {0: '2005Q3', 
                               1: '2009Q1', 
                               2: '2009Q3', 
                               3: '2010Q2'}
        
        self.quarters_total = ['1994Q1', '1994Q2', '1994Q3', '1994Q4', '1995Q1', '1995Q2', '1995Q3', '1995Q4', 
                               '1996Q1', '1996Q2', '1996Q3', '1996Q4', '1997Q1', '1997Q2', '1997Q3', '1997Q4',
                               '1998Q1', '1998Q2', '1998Q3', '1998Q4', '1999Q1', '1999Q2', '1999Q3', '1999Q4',
                               '2000Q1', '2000Q2', '2000Q3', '2000Q4', '2001Q1', '2001Q2', '2001Q3', '2001Q4',
                               '2002Q1', '2002Q2', '2002Q3', '2002Q4', '2003Q1', '2003Q2', '2003Q3', '2003Q4',
                               '2004Q1', '2004Q2', '2004Q3', '2004Q4', '2005Q1', '2005Q2', '2005Q3', '2005Q4',
                               '2006Q1', '2006Q2', '2006Q3', '2006Q4', '2007Q1', '2007Q2', '2007Q3', '2007Q4',
                               '2008Q1', '2008Q2', '2008Q3', '2008Q4', '2009Q1', '2009Q2', '2009Q3', '2009Q4',
                               '2010Q1', '2010Q2', '2010Q3', '2010Q4', '2011Q1', '2011Q2', '2011Q3', '2011Q4',
                               '2012Q1', '2012Q2', '2012Q3', '2012Q4', '2013Q1', '2013Q2', '2013Q3', '2013Q4',
                               '2014Q1', '2014Q2', '2014Q3', '2014Q4', '2015Q1', '2015Q2', '2015Q3']
        
        self.fund_col_names = ['niq', 'ltq', 'piq', 'atq', 'gind',
                               'ggroup', 'gsector', 'gsubind', 'lt_rating']

        self.index_col_names = ['ggroup', 'gsector', 'gsubind', 'gind', 'form_type', 'cik'] 

        self.macro_col_names = ['exalus', 'excaus', 'exchus', 'exjpus',
                                'exmxus', 'exszus','exukus', 'twexm',
                                'FF_O', 'PRIME_NA', 'TCMNOM_M6', 'TCMNOM_Y10',
                                'TCMNOM_Y1', 'TCMNOM_Y2', 'TCMNOM_Y5']

        self.metadata_cols = ['cik', 'form_type', 'change', 'gind', 'ggroup', 'gsector',
                              'gsubind', 'quarter', 'quarter_dt', 'lt_rating_0']

        self.text_col_names = ['mda']

        self.target_col = 'change'

        self.feature_importances_by_category = {'fundamental': 0, 'macro': 0, 'text': 0}
        self.feature_importances_by_category_accumulated = {'fundamental': [], 'macro': [], 'text': []}
        self.individual_feature_importances = {}
        
        if args.no_text:
            self.results_dir = f'results_static/no_text/{args.no_text_features}'
        elif args.only_text:
            self.results_dir = f'results_static/only_text/{self.feature_type}'
        else:
            self.results_dir = f'results_static/all/{self.feature_type}'

        os.makedirs(self.results_dir, exist_ok=True)

    def reset_metrics(self):
        self.accuracies = []
        self.f1_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.feature_importances_by_category_accumulated = {'fundamental': [], 'macro': [], 'text': []}
        self.feature_importances_by_category = {'fundamental': 0, 'macro': 0, 'text': 0}
        self.individual_feature_importances_accumulated = {}
        self.individual_feature_importances = {}

    def load_and_prepare_data(self, file_path):
        df = pd.read_csv(file_path)
        df['quarter'] = df['quarter'].astype(str)  
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'quarter':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except ValueError:
                    df[col] = df[col].astype('category')
        return df

    def relabel_classes(self, y):
        label_mapping = {-1: 0, 0: 1, 1: 2}
        return y.replace(label_mapping)
    
    def format_dataframe(self, df, lag):
        df = df.drop(columns=[self.target_col, 'quarter', 'quarter_dt'] + self.index_col_names, errors='ignore')
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        df = df.drop(columns=[f"mda_{i}" for i in range(lag+1)], errors='ignore')
        
        if args.no_text:
            if args.no_text_features == "all":
                keep_columns = [col for col in df.columns if extract_text_before_last_underscore(col) in self.fund_col_names + self.macro_col_names]
            elif args.no_text_features == "macro":
                keep_columns = [col for col in df.columns if extract_text_before_last_underscore(col) in self.macro_col_names]
            elif args.no_text_features == "fund":
                keep_columns = [col for col in df.columns if extract_text_before_last_underscore(col) in self.fund_col_names]
            elif args.no_text_features == "cr":
                keep_columns = [col for col in df.columns if extract_text_before_last_underscore(col) in ['lt_rating']]
            else:
                ValueError("Invalid variable name. 'no_text_features' variable needs to be equal to one of the following: 'all', 'macro', 'fund', or 'cr'.")

            df = df[keep_columns]
        elif args.only_text:
            non_text_col_names = self.fund_col_names + self.macro_col_names
            df = df.drop(columns=[col for col in df.columns if extract_text_before_last_underscore(col) in non_text_col_names], errors='ignore')
        else:
            pass
        return df

    def is_in_category(self, col):
        if col in self.index_col_names:
            print("INDEX!!", col)
            return 'indexing'
        elif any(col.startswith(base_name + "_") for base_name in self.fund_col_names):
            return 'fundamental'
        elif any(col.startswith(base_name + "_") for base_name in self.macro_col_names):
            return 'macro'
        else:
            return 'text'
    
    def aggregate_feature_importances(self, model, feature_names):
        for i, feature in enumerate(feature_names):
            importance = model.feature_importances_[i]
            category = self.is_in_category(feature)
            self.feature_importances_by_category[category] += importance
            self.individual_feature_importances[feature] = importance

    def train_and_evaluate(self, train_df, test_df, lag):
        X_train = self.format_dataframe(train_df, lag)
        y_train = self.relabel_classes(train_df[self.target_col])
        X_test = self.format_dataframe(test_df, lag)
        y_test = self.relabel_classes(test_df[self.target_col])
        
        metadata_test = test_df[self.metadata_cols]  

        model = xgb.XGBClassifier(objective="multi:softprob", random_state=42, use_label_encoder=True)
        model.fit(X_train, y_train)

        relevant_features = [col for col in X_train.columns if self.is_in_category(col) != 'indexing']
        self.aggregate_feature_importances(model, relevant_features)

        for key, value in self.feature_importances_by_category.items():
            if key in self.feature_importances_by_category_accumulated:
                self.feature_importances_by_category_accumulated[key].append(value)
            else:
                self.feature_importances_by_category_accumulated[key] = [value]

        for key, value in self.individual_feature_importances.items():
            if key in self.individual_feature_importances_accumulated:
                self.individual_feature_importances_accumulated[key].append(value)
            else:
                self.individual_feature_importances_accumulated[key] = [value]

        predictions = model.predict(X_test)
        self.evaluate_model(y_test, predictions)

        self.save_predictions_with_metadata(metadata_test, y_test, predictions, lag)
        # self.analyze_feature_gradients(model, X_train, y_train, lag)  


        return model, classification_report(y_test, predictions, output_dict=True)
    
    def analyze_feature_gradients(self, model, X, y, lag):

        classes = np.unique(y)  
        positive_class = np.where(classes == 2.0)[0][0] 
        negative_class = np.where(classes == 0.0)[0][0]  

        max_positive_gradient = {'feature': None, 'gradient': -np.inf}
        max_negative_gradient = {'feature': None, 'gradient': -np.inf}
        text_features = [col for col in X.columns if self.is_in_category(col) == 'text']

        for feature in tqdm(text_features, desc="Analyzing PDPs"):
            results = partial_dependence(model, X, features=[feature], kind='average')
            pdp_values = results['average'][0]
            mean_pdp = np.mean(pdp_values)
            
            gradient_change = np.max(pdp_values) - np.min(pdp_values)
            if mean_pdp > 0 and gradient_change > max_positive_gradient['gradient']:
                max_positive_gradient = {'feature': feature, 'gradient': gradient_change}
            elif mean_pdp > 0 and abs(gradient_change) > max_negative_gradient['gradient']:
                max_negative_gradient = {'feature': feature, 'gradient': abs(gradient_change)}

        fig, axs = plt.subplots(2, 1, figsize=(8, 12))

        font_size = 14 
        plt.rc('font', size=font_size)

        if max_positive_gradient['feature'] is not None:
            print("P")
            PartialDependenceDisplay.from_estimator(
                model, X, features=[max_positive_gradient['feature']], ax=axs[0], kind='average', target=positive_class
            )
            # axs[0].set_title(f"PDP of {max_positive_gradient['feature']} on 'Up' class")
            pos_text = str(max_positive_gradient['feature'])
            axs[0].set_xlabel(pos_text)
            axs[0].set_ylabel('Predicted outcome change')

        if max_negative_gradient['feature'] is not None:
            print("N")
            PartialDependenceDisplay.from_estimator(
                model, X, features=[max_negative_gradient['feature']], ax=axs[1], kind='average', target=negative_class
            )
            # axs[1].set_title(f"PDP of {max_negative_gradient['feature']} on 'Down' class")
            neg_text = str(max_negative_gradient['feature'])
            axs[1].set_xlabel(neg_text)
            axs[1].set_ylabel('Predicted outcome change')

        plt.tight_layout()
        plt.savefig(f'feature_gradients_{lag}.png')
        plt.show()


    def train_and_evaluate_proba(self, train_df, test_df, lag):
        X_train = self.format_dataframe(train_df, lag)
        y_train = self.relabel_classes(train_df[self.target_col])
        X_test = self.format_dataframe(test_df, lag)
        y_test = self.relabel_classes(test_df[self.target_col])
        
        metadata_test = test_df[self.metadata_cols]

        model = xgb.XGBClassifier(objective="multi:softprob", random_state=42, use_label_encoder=True, enable_categorical=True)
        model.fit(X_train, y_train)

        relevant_features = [col for col in X_train.columns if self.is_in_category(col) != 'indexing']
        self.aggregate_feature_importances(model, relevant_features)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        self.evaluate_model(y_test, predictions)
        self.save_predictions_with_metadata_proba(metadata_test, y_test, predictions, probabilities, lag)

        return model, predictions, probabilities, classification_report(y_test, predictions, output_dict=True)

    def save_predictions_with_metadata_proba(self, metadata, y_true, y_pred, y_prob, lag):
        results = metadata.copy()
        results['true_label'] = y_true
        results['predicted_label'] = y_pred
        results['predicted_probability'] = np.max(y_prob, axis=1)
        
        results_dir = f"{self.results_dir}/lag{lag}"
        os.makedirs(results_dir, exist_ok=True) 
        
        results_path = f"{results_dir}/predictions_with_metadata_proba_val.csv"
        
        if os.path.exists(results_path):
            existing_results = pd.read_csv(results_path)
            combined_results = pd.concat([existing_results, results]).drop_duplicates().reset_index(drop=True)
            combined_results.to_csv(results_path, index=False)
        else:
            results.to_csv(results_path, mode='w', header=True, index=False)

    def evaluate_model(self, y_true, y_pred):
        self.accuracies.append(accuracy_score(y_true, y_pred))
        self.f1_scores.append(f1_score(y_true, y_pred, average='weighted'))
        self.precision_scores.append(precision_score(y_true, y_pred, average='weighted'))
        self.recall_scores.append(recall_score(y_true, y_pred, average='weighted'))

    def calculate_and_save_average_metrics(self, lag):
        avg_accuracy = np.mean(self.accuracies)
        avg_f1 = np.mean(self.f1_scores)
        avg_precision = np.mean(self.precision_scores)
        avg_recall = np.mean(self.recall_scores)
        
        std_accuracy = np.std(self.accuracies)
        std_f1 = np.std(self.f1_scores)
        std_precision = np.std(self.precision_scores)
        std_recall = np.std(self.recall_scores)
        
        metrics = {
            'accuracy': {'mean': avg_accuracy, 'std': std_accuracy},
            'f1_score': {'mean': avg_f1, 'std': std_f1},
            'precision': {'mean': avg_precision, 'std': std_precision},
            'recall': {'mean': avg_recall, 'std': std_recall}
        }
        
        metrics_path = f'{self.results_dir}/lag{lag}/averaged_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Averaged Metrics for Lag {lag}:")
        for metric, values in metrics.items():
            print(f"{metric}: {values['mean']:.4f} (±{values['std']:.4f})")
    
    def save_results(self, model, classif_report, feature_type, lag, quarter):
        directory = f'{self.results_dir}/lag{lag}/rolling'
        models_directory = f'{directory}/models'  
        reports_directory = f'{directory}/classification_reports' 
        importances_directory = f'{directory}/importances' 
        
        os.makedirs(directory, exist_ok=True)
        os.makedirs(models_directory, exist_ok=True)
        os.makedirs(reports_directory, exist_ok=True)
        os.makedirs(importances_directory, exist_ok=True)

        joblib.dump(model, f'{models_directory}/trained_to_{quarter}.pkl')
        
        with open(f'{reports_directory}/classification_report_{quarter}.txt', 'w') as f:
            f.write(str(classif_report))

        self.save_feature_importances(importances_directory, quarter)

    def save_feature_importances(self, importances_directory, quarter):

        cat_directory = f'{importances_directory}/categories' 
        ind_directory = f'{importances_directory}/individual' 
        
        os.makedirs(cat_directory, exist_ok=True)
        os.makedirs(ind_directory, exist_ok=True)
        
        feature_importances_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                            for k, v in self.feature_importances_by_category.items()}
        with open(f'{cat_directory}/{quarter}.json', 'w') as f:
            json.dump(feature_importances_serializable, f, indent=4)

        individual_importances_serializable = {
            k: (float(v) if isinstance(v, np.floating) else v) for k, v in self.individual_feature_importances.items()
        }
        
        with open(f'{ind_directory}/{quarter}.json', 'w') as f:
            json.dump(individual_importances_serializable, f, indent=4)

    def average_feature_importances(self):
        averaged_by_category = {k: np.mean(v) for k, v in self.feature_importances_by_category_accumulated.items()}
        averaged_individual = {k: np.mean(v) for k, v in self.individual_feature_importances_accumulated.items()}
        return averaged_by_category, averaged_individual

    def save_averaged_feature_importances(self):
        averaged_by_category, averaged_individual = self.average_feature_importances()
        
        with open(f'{self.results_dir}/averaged_total_feature_importances_by_category.json', 'w') as f:
            json.dump(averaged_by_category, f, indent=4)
        
        with open(f'{self.results_dir}/averaged_individual_feature_importances.json', 'w') as f:
            json.dump(averaged_individual, f, indent=4)

    def save_grouped_accuracy(self, results_df, lag, group_column, file_name):
        
        grouped_data = results_df.groupby(group_column).apply(
            lambda x: pd.Series({
                'accuracy': (x['true_label'] == x['predicted_label']).mean(),
                'count': len(x)
            })
        )
        accuracy_path = f"{self.results_dir}/lag{lag}/{file_name}.csv"
        os.makedirs(os.path.dirname(accuracy_path), exist_ok=True)  
        grouped_data.to_csv(accuracy_path, header=True)

    def save_accuracies(self, results_df, lag):

        self.save_grouped_accuracy(results_df, lag, 'lt_rating_0', 'accuracy_by_rating')
        self.save_grouped_accuracy(results_df, lag, 'gind', 'accuracy_by_gind')
        self.save_grouped_accuracy(results_df, lag, 'quarter', 'accuracy_by_quarter')
        self.save_grouped_accuracy(results_df, lag, 'change', 'accuracy_by_change')


    def save_predictions_with_metadata(self, metadata, y_true, y_pred, lag):
        results = metadata.copy()
        results['true_label'] = y_true
        results['predicted_label'] = y_pred
        
        results_dir = f"{self.results_dir}/lag{lag}"
        os.makedirs(results_dir, exist_ok=True) 
        
        results_path = f"{results_dir}/predictions_with_metadata.csv"
        
        if os.path.exists(results_path):
            existing_results = pd.read_csv(results_path)
            combined_results = pd.concat([existing_results, results]).drop_duplicates().reset_index(drop=True)
            combined_results.to_csv(results_path, index=False)
        else:
            results.to_csv(results_path, mode='w', header=True, index=False)

    def main(self):
        feature_types = ["bert_emo", "bert", "clusters_all-mpnet-base-v2", "lda", "lm_lex",
                         "longformer", "nrc_lex", "tfidf_unigrams", "clusters_all-roberta-large-v1", "clusters"]
        feature_types = ['gpt4o_text_only']
        feature_types = ["clusters_all-roberta-large-v1"]
        for feature_type in feature_types:
            self.feature_type = feature_type
            if args.no_text:
                self.results_dir = f'results_static/no_text/{args.no_text_features}'
            elif args.only_text:
                self.results_dir = f'results_static/only_text/{self.feature_type}'
            else:
                self.results_dir = f'results_static/all/{self.feature_type}'
            print("Feature type", self.feature_type)
            for lag in range(4):  
                self.reset_metrics()  
                file_path = f"dataframes/lag{lag}/{self.feature_type}.csv"
                df = self.load_and_prepare_data(file_path)

                train_df = df[df['quarter'].str[:4].astype(int) <= 2012]
                test_df = df[df['quarter'].str[:4].astype(int).isin([2015, 2016])]

                if not test_df.empty:
                    model, classif_report = self.train_and_evaluate(train_df, test_df, lag)
                    self.save_results(model, classif_report, self.feature_type, lag, "2012Q4")                  

                    results_path = f"{self.results_dir}/lag{lag}/predictions_with_metadata.csv"
                    results_df = pd.read_csv(results_path)
                    self.save_accuracies(results_df, lag)
  
                averaged_by_category, averaged_individual = self.average_feature_importances()
                self.save_averaged_feature_importances_for_lag(averaged_by_category, averaged_individual, lag)
                self.calculate_and_save_average_metrics(lag)

                self.feature_importances_by_category_accumulated = {'fundamental': [], 'macro': [], 'text': []}
                self.individual_feature_importances_accumulated = {}

    def save_averaged_feature_importances_for_lag(self, averaged_by_category, averaged_individual, lag):
        with open(f'{self.results_dir}/lag{lag}/averaged_total_feature_importances_by_category.json', 'w') as f:
            json.dump(averaged_by_category, f, indent=4)
        
        averaged_individual = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in averaged_individual.items()}

        with open(f'{self.results_dir}/lag{lag}/averaged_individual_feature_importances.json', 'w') as f:
            json.dump(averaged_individual, f, indent=4)

def convert(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError

if __name__ == "__main__":
    trainer = ModelTrainer(args)
    trainer.main()