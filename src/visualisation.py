import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

labels_list = ['ORG', 'PER', 'LOC', 'MISC']


def plot_categ_distribution(data):
    labels_dict = dict(zip(labels_list, [0] * 4))

    def count_labels(tokens):
        for token in tokens:
            if token.startswith('B'):
                labels_dict[token.split('-')[-1]] += 1

    data['labels'].apply(count_labels)
    plt.bar(labels_dict.keys(), labels_dict.values())


def visualize_metrics(category_metrics, overall_metrics):
    categories = list(category_metrics.keys())
    precisions = [category_metrics[cat]['precision'] for cat in categories]
    recalls = [category_metrics[cat]['recall'] for cat in categories]
    f1_scores = [category_metrics[cat]['f1'] for cat in categories]

    bar_width = 0.25
    r1 = np.arange(len(categories))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(12, 6))
    plt.bar(r1, precisions, color='skyblue', width=bar_width, edgecolor='grey', label='Precision')
    plt.bar(r2, recalls, color='lightgreen', width=bar_width, edgecolor='grey', label='Recall')
    plt.bar(r3, f1_scores, color='salmon', width=bar_width, edgecolor='grey', label='F1 Score')

    plt.axhline(overall_metrics['overall_precision'], color='blue', linewidth=1.5, linestyle='--', label='Overall Precision')
    plt.axhline(overall_metrics['overall_recall'], color='green', linewidth=1.5, linestyle='--', label='Overall Recall')
    plt.axhline(overall_metrics['overall_f1'], color='red', linewidth=1.5, linestyle='--', label='Overall F1 Score')

    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Category-Specific and Overall Metrics', fontsize=16)
    plt.xticks([r + bar_width for r in range(len(categories))], categories)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_cooccurrence(data: pd.DataFrame) -> None:
    co_occurrence = pd.DataFrame(0, index=labels_list, columns=labels_list)

    for _, row in data.iterrows():
        sentence_labels = set(label.split('-')[-1] for label in row['labels'] if label != 'O')
        for cat1 in sentence_labels:
            for cat2 in sentence_labels:
                if cat1 == cat2:
                    continue
                if cat1 in labels_list and cat2 in labels_list:
                    co_occurrence.loc[cat1, cat2] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(co_occurrence, annot=True, cmap="YlGnBu", fmt="d", square=True,
                cbar_kws={'label': 'Number of Co-occurrences'}, linewidths=0.5)
    plt.title("NER Category Co-occurrence Heatmap", fontsize=16)
    plt.xlabel("NER Category", fontsize=14)
    plt.ylabel("NER Category", fontsize=14)
    plt.tight_layout()
    plt.show()


def draw_confusion_matrix(predictions, labels):
    flatten_tag = lambda tag: tag if tag == 'O' else tag.split('-')[-1]
    flattened_true = [flatten_tag(tag) for sentence in labels for tag in sentence]
    flattened_pred = [flatten_tag(tag) for sentence in predictions for tag in sentence]

    labels = sorted(set(flattened_true) | set(flattened_pred))
    labels.remove('O')
    conf_matrix = confusion_matrix(flattened_true, flattened_pred, labels=labels)

    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix for NER')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
