import logging
import sys

import networkx as nx
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# sys.path.insert(0, 'components/')
from model_file import LinkPredict
from arguments import solicit_params
from data_process import text_to_assertions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


def load_data(path, filename, limit=-1):
    pathname = path + filename
    assertions = list(set(text_to_assertions(pathname, n=limit, concept_filter=[])))
    return assertions


def create_graph(samples):
    G = nx.Graph()
    vocabulary = set()
    edge_types = set()
    for item in samples:
        vocabulary.add(item.subject)
        vocabulary.add(item.object)
        G.add_edge(item.subject, item.object, weight=1, rel_type=item.relation, norm=1)
        edge_types.add(item.relation)
    vocabulary, edge_types = list(vocabulary), list(edge_types)

    logging.debug("Adding all relations")
    d = []
    for item in tqdm(samples):
        d.append(tuple([vocabulary.index(item.subject),
                        edge_types.index(item.relation),
                        vocabulary.index(item.object)]))
    logging.debug("Making the set")
    d = list(set(d))
    logging.debug("Restoring list back...")
    d = np.array([list(x) for x in d])
    return G, edge_types, vocabulary, d


if __name__ == "__main__":
    args = solicit_params()
    path = args.data_path
    filename = args.filename
    data_limit = args.limit
    if args.dev_run:
        data_limit = 7000
        args.n_gpus = 0

    eval_protocol = args.eval_protocol
    lr = args.lr
    samples = load_data(path, filename, data_limit)
    test_samples = load_data(path, args.test_filename)

    G, edge_types, vocabulary, d = create_graph(samples)
    G_test, edge_types_test, vocabulary_test, test_data = create_graph(test_samples)

    all_vocabulary = set(vocabulary + vocabulary_test)
    all_edge_types = set(edge_types + edge_types_test)

    vocab = torch.nn.Embedding(len(all_vocabulary), args.embedding_dim)
    num_nodes = vocab.weight.shape[0]
    num_rels = len(all_edge_types)
    n_hidden = num_rels * 4
    in_dim = vocab.weight.shape[0]

    n_bases = args.n_bases
    n_layers = args.n_layers
    dropout = args.dropout
    regularization = args.regularization

    train_data, valid_data = train_test_split(d, test_size=0.10, random_state=1)

    wandb_logger = WandbLogger(project='kgg_lit', name=args.output_dir, config=args)

    train_dataset = RelationsData(num_nodes, num_rels, train_data)
    valid_dataset = RelationsData(num_nodes, num_rels, valid_data, n_data=1, is_val=True)
    test_dataset = RelationsData(num_nodes, num_rels, test_data, n_data=1, is_test=True)

    def custom_collate_fn(items):
        return items[0]
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    eval_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate_fn)

    if args.dev_run:
        trainer = Trainer(fast_dev_run=True)
    else:
        trainer = Trainer(devices=args.n_gpus,
                          accelerator="gpu" if torch.cuda.is_available() else "cpu",
                          strategy='ddp' if torch.cuda.is_available() else None,
                          logger=wandb_logger,
                          val_check_interval=1.0,
                          max_epochs=args.epochs)

    model = LinkPredict(train_data,
                           valid_data,
                           test_data,
                           in_dim,
                           n_hidden,
                           num_rels,
                           eval_batch_size=args.eval_batch_size,
                           eval_protocol=eval_protocol,
                           num_bases=n_bases,
                           num_hidden_layers=n_layers,
                           dropout=dropout,
                           reg_param=regularization,
                           vocabulary=all_vocabulary,
                           embedding_dim=args.embedding_dim
                           )

    trainer.fit(model, train_loader, eval_loader)
    trainer.test(model=model, test_dataloaders=test_loader)
