import os
from argparse import ArgumentParser

import pandas as pd

from scripts.preprocessing.preprocessing_parameters import REPLACE_AMP_MAP, EMOJI_MAPS_MAP

from scripts.preprocessing.preprocessing_utils import preprocess_tweet_text


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', default=r"../../data/raw/ru")
    parser.add_argument('--lang', default="ru")
    parser.add_argument('--output_dir', default=r"../../data/preprocessed/ru")
    args = parser.parse_args()

    input_dir = args.input_dir
    language = args.lang
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    amp_replace = REPLACE_AMP_MAP[language]
    emoji_mapping = EMOJI_MAPS_MAP[language]

    for filename in os.listdir(input_dir):
        dataset_type = filename.split('.')[0]
        if dataset_type == 'train' or dataset_type == 'dev':
            columns = ["class", "tweet"]
        elif dataset_type == 'test':
            columns = ["tweet_id", "tweet"]
        else:
            raise Exception(f"Invalid filename: {filename}")
        input_path = os.path.join(input_dir, filename)
        data_df = pd.read_csv(input_path, sep="\t", encoding="utf-8")[columns]
        data_df['tweet'] = data_df['tweet'].apply(lambda x: preprocess_tweet_text(x, emoji_mapping, amp_replace))
        output_path = os.path.join(output_dir, filename)
        data_df.to_csv(output_path, encoding="UTF-8", sep="\t", index=False, header=None)


if __name__ == '__main__':
    main()
