from argparse import ArgumentParser
import os
import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_files', nargs='+', default=[r"../data/preprocessed/ru/train.tsv",
                                                             r"../data/preprocessed/en/train.tsv"])
    parser.add_argument('--output_path', default=r"../data/preprocessed/ruen/train.tsv")
    args = parser.parse_args()

    random_state = 42
    input_paths_list = args.input_files
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    dataframes_list = []
    for input_path in input_paths_list:
        filename = os.path.basename(input_path)
        print(input_path)
        dataset_type = filename.split('.')[0]
        if dataset_type == 'train' or dataset_type == 'dev':
            columns = ["class", "tweet"]
        elif dataset_type == 'test':
            columns = ["tweet_id", "tweet"]
        else:
            raise Exception(f"Invalid filename: {filename}")
        data_df = pd.read_csv(input_path, sep="\t", encoding="utf-8", header=None, names=columns)[columns]
        print(data_df.shape)
        dataframes_list.append(data_df)
    result_df = pd.concat(dataframes_list).sample(frac=1, random_state=random_state)
    print("Result shape", result_df.shape)

    result_df.to_csv(output_path, encoding="UTF-8", sep="\t", index=False, header=None)


if __name__ == '__main__':
    main()
