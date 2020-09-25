import os
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = ['sentences', 'EF', 'INF', 'ADR', 'DI', 'Finding']


def main():
    parser = ArgumentParser()
    parser.add_argument('--psytar_dir', default=r"../../med_reviews_corpora/psytar_csvs/")
    parser.add_argument('--rudrec_dir', default=r"../../med_reviews_corpora/otzovik_csvs/fold_0")
    parser.add_argument('--output_dir', default=r"../../med_reviews_corpora/merged_reviews_kfolds")
    args = parser.parse_args()

    psytar_dir = args.psytar_dir
    otzovik_dir = args.otzovik_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    random_state = 42

    otzovik_train_df = pd.read_csv(os.path.join(otzovik_dir, "train.csv"), encoding="utf-8")[COLUMNS]
    otzovik_test_df = pd.read_csv(os.path.join(otzovik_dir, "test.csv"), encoding="utf-8")[COLUMNS]
    otzovik_dev_df = pd.read_csv(os.path.join(otzovik_dir, "dev.csv"), encoding="utf-8")[COLUMNS]

    psytar_train_df = pd.read_csv(os.path.join(psytar_dir, "train.csv"), encoding="utf-8")[COLUMNS]
    psytar_test_df = pd.read_csv(os.path.join(psytar_dir, "test.csv"), encoding="utf-8")[COLUMNS]
    psytar_dev_df = pd.read_csv(os.path.join(psytar_dir, "dev.csv"), encoding="utf-8")[COLUMNS]

    merged_train_df = pd.concat([psytar_train_df, otzovik_train_df, psytar_test_df, otzovik_test_df]) \
        .sample(frac=1, random_state=random_state)
    # merged_test_df = pd.concat([psytar_test_df, otzovik_test_df]).sample(frac=1, random_state=random_state)
    merged_dev_df = pd.concat([psytar_dev_df, otzovik_dev_df]).sample(frac=1, random_state=random_state)

    merged_train_df.to_csv(os.path.join(output_dir, "train.csv"), encoding="utf-8", index=False)
    # merged_test_df.to_csv(os.path.join(output_dir, "test.csv"), encoding="utf-8", index=False)
    merged_dev_df.to_csv(os.path.join(output_dir, "dev.csv"), encoding="utf-8", index=False)


if __name__ == '__main__':
    main()
