from collections import defaultdict
import spacy
from spacy.tokens import Doc, DocBin
import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

labels_list = ['ORG', 'PER', 'LOC', 'MISC']
labels_set = list(labels_list)
global_metrics = ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']

labels_correspondance = {
    'ORG': 'ORG',
    'PERSON': 'PER',
    'LOC': 'LOC',
    'GPE': 'LOC',
    'FAC': 'LOC',
    'PRODUCT': 'MISC',
    'EVENT': 'MISC',
    'WORK_OF_ART': 'MISC',
    'LANGUAGE': 'O',
    'DATE': 'O',
    'TIME': 'O',
    'PERCENT': 'O',
    'MONEY': 'O',
    'QUANTITY': 'O',
    'ORDINAL': 'O',
    'CARDINAL': 'O'
}


def convert_tag(tag):
    if tag == 'O':
        return tag
    bio, category = tag.split('-')
    new_category = labels_correspondance.get(category, 'O')
    if new_category == 'O':
        return new_category
    return f'{bio}-{new_category}'


def calculate_metrics(predictions: list[str], labels: list[str], metric):
    metrics = metric.compute(predictions=list(predictions), references=labels)
    metrics_class = {k: metrics[k] for k in labels_list}
    metrics_global = {k: metrics[k] for k in global_metrics}
    print(pd.DataFrame(metrics_class), '\n')
    print(pd.DataFrame([metrics_global], columns=list(metrics_global.keys()), index=['Value']))
    return metrics_class, metrics_global


def compute_spacy_metrics(docs, labels, metric):
    predictions = map(lambda row: [e.ent_iob_ + '-' + e.ent_type_ if e.ent_iob_ != 'O' else 'O' for e in row], docs)
    predictions = list(map(lambda row: [convert_tag(e) for e in row], predictions))
    return calculate_metrics(predictions, labels, metric)


def make_docs(data, nlp, save_path):
    predictions = data['tokens'].progress_apply(lambda row: nlp(Doc(nlp.vocab, row)))
    doc_bin = DocBin(store_user_data=True)
    for doc in predictions:
        doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()
    with open(save_path, 'wb') as fout:
        fout.write(bytes_data)
    return bytes_data


def load_docs(save_path, nlp):
    with open(save_path, 'rb') as fout:
        bytes_data = fout.read()
    doc_bin = DocBin().from_bytes(bytes_data)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return docs


def extract_pos_distribution(data: pd.DataFrame,
                             preds: list[spacy.tokens.Doc]) -> defaultdict:
    pos_distribution = defaultdict(lambda: defaultdict(int))
    pos_tags = [[token.pos_ for token in document] for document in preds]
    for idx, row in data.iterrows():
        for token, label, pos in zip(row['tokens'], row['labels'], pos_tags[idx]):
            if label != 'O':
                entity_type = label.split('-')[-1]
                pos_distribution[entity_type][pos] += 1

    return pos_distribution


def visualize_pos_distribution(pos_distribution: defaultdict) -> None:
    pos_df = pd.DataFrame(pos_distribution).fillna(0).T.astype(int)
    plt.figure(figsize=(14, 10))
    sns.heatmap(pos_df, annot=True, fmt="d", cmap="Blues",
                cbar_kws={'label': 'Frequency'}, linewidths=0.5,
                annot_kws={"size": 10})

    plt.title("PoS Distribution for each NER Category", fontsize=16)
    plt.ylabel("NER Categories", fontsize=14)
    plt.xlabel("POS Tags", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def extract_predictions(data: pd.DataFrame, preds: list[spacy.tokens.Doc]) -> pd.DataFrame:
    data['raw_predictions'] = list(map(lambda row: [e.ent_iob_ + '-' + e.ent_type_ if e.ent_iob_ != 'O' else 'O' for e in row], preds))
    data['predictions'] = list(map(lambda row: [convert_tag(e) for e in row], data['raw_predictions']))
    return data


def error_example(data: pd.DataFrame, category: str, n_examples: int = 5):
    true = data.labels.apply(lambda x: f'B-{category}' in x)
    positive = data.predictions.apply(lambda x: f'B-{category}' in x)

    for title, examples in zip(
        [f'Unrecognized {category} entities',
         f'Falsely recognized  {category} entities'],
        [data[true & ~positive], data[~true & positive]]
    ):
        print(f'\033[1m{title}\033[0;0m:\n')
        examples = examples[:n_examples]
        for example in range(min(n_examples, len(examples))):
              print(
                pd.DataFrame(
                    {
                        'tokens:': examples.tokens.iloc[example],
                        'true  :': examples.labels.iloc[example],
                        'pred  :': examples.predictions.iloc[example],
                        'raw   :': examples.raw_predictions.iloc[example]

                    }
                )
                .transpose()
                .to_string(index=True, header=False),
                '\n'
            )
