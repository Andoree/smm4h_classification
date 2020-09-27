import os
from argparse import ArgumentParser

import modeling
import pandas as pd
import tensorflow as tf
import tokenization
from bert_preprocessing import create_examples
from multilabel_bert import file_based_input_fn_builder, create_model, model_fn_builder, \
input_fn_builder, create_output, predict, get_estimator, train_and_evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    parser = ArgumentParser()
    parser.add_argument('--corpus_dir', )
    parser.add_argument('--bert_vocab', )
    parser.add_argument('--bert_checkpoint', )
    parser.add_argument('--bert_config', )
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warmup_proportion', type=float)
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--save_summary_steps', type=int)
    parser.add_argument('--text_column', )
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir',)
    parser.add_argument('--prediction_filename', )
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    bert_vocab_path = args.bert_vocab
    bert_init_chkpnt_path = args.bert_checkpoint
    bert_config_path = args.bert_config
    batch_size = args.batch_size
    num_train_epochs = args.epochs
    warmup_proportion = args.warmup_proportion
    max_seq_length = args.max_seq_length
    learning_rate = args.learning_rate
    save_summary_steps = args.save_summary_steps
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    predicted_proba_filename = args.prediction_filename
    # Number of classes
    num_labels = args.num_labels
    # The column with this name must exist in test data.
    text_column_name = args.text_column

    # Change paths if needed
    train_df = pd.read_csv(os.path.join(corpus_dir, "train.csv"), encoding="utf-8")
    dev_df = pd.read_csv(os.path.join(corpus_dir, "dev.csv"), encoding="utf-8")

    train_examples = create_examples(train_df)
    eval_examples = create_examples(dev_df)
    # Model is saved and evaluated every epoch. It might be too frequent, change it.
    num_train_steps = int(len(train_examples) / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    num_steps_in_epoch = int(len(train_examples) / batch_size * num_train_epochs) // num_train_epochs
    save_checkpoints_steps = num_steps_in_epoch

    # Creating tokenizer
    tokenizer = tokenization.FullTokenizer(
        vocab_file=bert_vocab_path, do_lower_case=True)
    # Definition of estimator's config
    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=save_summary_steps,
        keep_checkpoint_max=1,
        save_checkpoints_steps=save_checkpoints_steps)
    # Loading config of pretrained Bert model
    bert_config = modeling.BertConfig.from_json_file(bert_config_path)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        init_checkpoint=bert_init_chkpnt_path,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = get_estimator(model_fn=model_fn, run_config=run_config, batch_size=batch_size)

    tf.logging.set_verbosity(tf.logging.INFO)
    eval_steps = None
    train_and_evaluate(train_examples, eval_examples, max_seq_length, estimator, tokenizer, batch_size, eval_steps,
                       num_train_steps, output_dir, num_labels=num_labels)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    save_checkpoints_steps = 1000

    # Creating tokenizer
    tokenizer = tokenization.FullTokenizer(
        vocab_file=bert_vocab_path, do_lower_case=True)
    # Definition of estimator's config
    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=save_summary_steps,
        keep_checkpoint_max=1,
        save_checkpoints_steps=save_checkpoints_steps)
    # Loading config of pretrained Bert model
    bert_config = modeling.BertConfig.from_json_file(bert_config_path)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        init_checkpoint=bert_init_chkpnt_path,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = get_estimator(model_fn=model_fn, run_config=run_config, batch_size=batch_size)

    test_df = pd.read_csv(os.path.join(corpus_dir, "dev.csv"), encoding="utf-8")
    label_names = {"p_label_1": "EF", "p_label_2": "INF", "p_label_3": "ADR", "p_label_4": "DI", "p_label_5": "Finding"}
    output_df = predict(test_df, estimator, tokenizer, max_seq_length, num_labels=num_labels)


    resulting_df = pd.concat([test_df, output_df], axis=1)
    resulting_df.to_csv(os.path.join(output_dir, predicted_proba_filename), index=False)
    resulting_df.rename(columns=label_names, inplace=True)
    METRICS = {"Precision": precision_score, "Recall": recall_score,
               "F-score": f1_score, }

    predicted_probs_pos_end = resulting_df.shape[1]
    predicted_probs_pos_start = predicted_probs_pos_end - num_labels
    columns = resulting_df.columns
    labels = columns[1: 1 + num_labels]
    results_numpy = resulting_df.values.transpose()
    all_true_labels = results_numpy[1: 1 + num_labels].astype(int)
    all_pred_probs = results_numpy[predicted_probs_pos_start: predicted_probs_pos_end]
    all_pred_labels = (all_pred_probs >= threshold).astype(int)
    print(f"Pretraining validation ({threshold} decision threshold):")
    for i in range(num_labels):
        class_true_labels = all_true_labels[i]
        class_pred_labels = all_pred_labels[i]
        label_name = labels[i]
        print(i, label_name)
        for metric_name, metric in METRICS.items():
            score = metric(y_true=class_true_labels, y_pred=class_pred_labels, labels=labels, )
            print(f"\t{metric_name} : {score}")

if __name__ == '__main__':
    main()
