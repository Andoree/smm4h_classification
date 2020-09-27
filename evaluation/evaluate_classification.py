import codecs
import os
from argparse import ArgumentParser

import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report

METRICS = {"Precision": precision_score, "Recall": recall_score,
           "F-score": f1_score, }


def main():
    parser = ArgumentParser()
    parser.add_argument('--true_labels_path', )
    parser.add_argument('--predicted_labels_path', )
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_path', )
    args = parser.parse_args()

    true_labels_path = args.true_labels_path
    predicted_labels_path = args.predicted_labels_path
    output_path = args.output_path
    threshold = args.threshold
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    true_labels_and_tweets_df = pd.read_csv(true_labels_path, sep="\t", quoting=3, quotechar=None, encoding="utf-8",
                                            header=None, names=["class", "tweet"])
    true_labels_df = true_labels_and_tweets_df["class"]

    predicted_labels_df = pd.read_csv(predicted_labels_path, sep="\t", encoding="utf-8", header=None)
    predicted_positive_probs_df = predicted_labels_df.iloc[:, 1]
    predicted_labels_df = predicted_positive_probs_df.apply(lambda x: 1 if x > threshold else 0)

    results = {}
    with codecs.open(output_path, "a+", encoding="utf-8") as output_file:
        print(classification_report(true_labels_df, predicted_labels_df))
        for metric_name, metric in METRICS.items():
            results[metric_name] = metric(true_labels_df, predicted_labels_df)
            print(f"{metric_name}", metric(true_labels_df, predicted_labels_df))
        output_file.write(",".join([str(x) for x in results.values()]))
        output_file.write('\n')


if __name__ == '__main__':
    main()
