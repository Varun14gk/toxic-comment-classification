#importing the libraries for data analyser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis


class DataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data['comment_text'].fillna("unknown", inplace=True)

    def display_head_tail_sample(self):
        print(self.data.head(1))
        #print(self.data.tail())
        #print(self.data.sample(n=10))

    def data_overview(self):
        print(self.data.shape)
        print(self.data.columns)
        print(self.data.info())
        #print(self.data.describe())
        print(self.data.isnull().sum())
        print(len(self.data))

    def comment_length_analysis(self):
        comment_len_data = self.data.comment_text.str.len()
        print(comment_len_data.describe())
        self.plot_comment_length_distribution(comment_len_data)


    def plot_comment_length_distribution(self, comment_len_data): 
        mean_length = comment_len_data.mean() 
        median_length = comment_len_data.median() 
        mode_length = comment_len_data.mode()[0] 
        skewness = skew(comment_len_data) 
        kurtosis_value = kurtosis(comment_len_data) 
        
        plt.figure(figsize=(10, 6)) 
        plt.hist(comment_len_data, bins=50, color='skyblue', edgecolor='black') 
        
        # Annotate mean 
        plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=1) 
        plt.text(mean_length, plt.ylim()[1]*0.95, f'Mean: {mean_length:.2f}', color='red', ha='right', fontweight='bold') 
        
        # Annotate median 
        plt.axvline(median_length, color='green', linestyle='dashed', linewidth=1) 
        plt.text(median_length, plt.ylim()[1]*0.85, f'Median: {median_length:.2f}', color='green', ha='right', fontweight='bold') 
        
        # Annotate mode 
        plt.axvline(mode_length, color='blue', linestyle='dashed', linewidth=1) 
        plt.text(mode_length, plt.ylim()[1]*0.75, f'Mode: {mode_length}', color='blue', ha='right', fontweight='bold') 
        
        # Annotate skewness 
        plt.text(plt.xlim()[1]*0.98, plt.ylim()[1]*0.80, f'Skewness: {skewness:.2f}', color='purple', ha='right', fontweight='bold') 
        
        # Annotate kurtosis 
        plt.text(plt.xlim()[1]*0.98, plt.ylim()[1]*0.70, f'Kurtosis: {kurtosis_value:.2f}', color='orange', ha='right', fontweight='bold') 
        
        # Add title and labels 
        plt.title('Distribution of Comment Lengths') 
        plt.xlabel('Length of Comments') 
        plt.ylabel('Frequency') 
        plt.grid(True) 
        
        # Show the plot 
        plt.show()

        
    def plot_label_distribution(self, column_name, title, labels):  
        plt.figure(figsize=(6, 4))  
        # Separate the data into two groups based on the value  
        data_0 = self.data[self.data[column_name] == 0]  
        data_1 = self.data[self.data[column_name] == 1] 
        
        # Get the counts for each group 
        count_0 = len(data_0) 
        count_1 = len(data_1) 
        
        # Plot each group with a different color  
        plt.hist(data_0[column_name], bins=np.arange(3) - 0.5, rwidth=0.5, color='green', edgecolor='black', label=f'{labels[0]} (n={count_0})')  
        plt.hist(data_1[column_name], bins=np.arange(3) - 0.5, rwidth=0.5, color='red', edgecolor='black', label=f'{labels[1]} (n={count_1})') 
        
        # Annotate counts 
        plt.text(0, count_0, f'{count_0}', color='black', ha='center', va='bottom') 
        plt.text(1, count_1, f'{count_1}', color='black', ha='center', va='bottom') 
        
        # Set the ticks, labels, and title  
        plt.xticks(ticks=[0, 1], labels=labels) 
        plt.title(title)  
        plt.xlabel(column_name) 
        plt.ylabel('Count') 
        plt.grid(axis='y') 
        
        # Add a legend to distinguish the groups 
        plt.legend() 
        
        # Show the plot 
        plt.show()

    def plot_correlation_heatmap(self):
        y_variables = self.data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
        corr_matrix = y_variables.corr()
        plt.figure(figsize=(8, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis', cbar=True, square=True)
        plt.title('Heatmap of Correlation Between Labels')
        plt.show()

# Usage
if __name__ == "__main__":
    analyzer = DataAnalyzer('data.csv')
    analyzer.display_head_tail_sample()
    analyzer.data_overview()
    analyzer.comment_length_analysis()
    analyzer.plot_label_distribution('toxic', 'Distribution of Toxic Comments', ['Not Toxic', 'Toxic'])
    analyzer.plot_label_distribution('severe_toxic', 'Distribution of Severe Toxic Comments', ['Not Severe Toxic', 'Severe Toxic'])
    analyzer.plot_label_distribution('obscene', 'Distribution of Obscene Comments', ['Not Obscene', 'Obscene'])
    analyzer.plot_label_distribution('threat', 'Distribution of Threat Comments', ['Not Threat', 'Threat'])
    analyzer.plot_label_distribution('insult', 'Distribution of Insult Comments', ['Not Insult', 'Insult'])
    analyzer.plot_label_distribution('identity_hate', 'Distribution of Identity Hate Comments', ['Not Identity Hate', 'Identity Hate'])
    analyzer.plot_correlation_heatmap()
