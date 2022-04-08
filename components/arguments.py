from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def solicit_params():
    parser = ArgumentParser()
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--model_state_file', type=str, default='model_state.pth')
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--tune', type=bool, default=False)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--sampling_method', type=str, default='uniform')
    # ----- DATASET -----
    parser.add_argument('--filename', type=str, default='conceptnet/train100k.txt')
    parser.add_argument('--test_filename', type=str, default='conceptnet/test.txt')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--filter', type=bool, default=False)
    parser.add_argument('--limit', type=int, default=-1)

    # ----- TRAINING AND EVALUATION -----
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--eval_protocol', type=str, default='filtered')
    parser.add_argument('--graph_split_size', type=float, default=0.5)
    parser.add_argument('--graph_batch_size', type=int, default=8192)

    # ----- PARAMETER OPTIMIZATION -----
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--regularization', type=float, default=0.0005)
    parser.add_argument('--n_bases', type=int, default=-1)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--hidden_dim', type=int, default=12)


    args = parser.parse_args()
    return args
