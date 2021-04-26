import sys

sys.path.insert(0, 'models/')
# Default Imports
import wandb
import json
import os
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# Data
from data import AskUbuntuDataModule
# Models
from models.outputadaptermodel import LitOutputAdapterBertModel
from models.outputmodel import LitOutputBertModel
from models.basemodel import LitBertModel
from models.inputmodel import LitInputBertModel

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = ArgumentParser(description='Train a model to do information retrieval on askubuntu dataset')

    parser.add_argument('--save_iters', type=int, default=5000,
                        help='The amount of steps to save a model')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The amount of epochs to run during training')
    parser.add_argument('--data_dir', default="./data/askubuntu/",
                        help='The root of the data directory for the askubuntu dataset')
    parser.add_argument('--model_dir', default="./models/",
                        help='The directory to save trained/checkpointed models')
    parser.add_argument('-cf', '--config_file',
                        help='The path to a json config file')
    parser.add_argument('--model_save_name', default="test_name",
                        help='The name to save the model with')
    parser.add_argument('--model_type', default=None,
                        help='The type of model that we use for evaluation')
    parser.add_argument('--gpus', default=None,
                        help='The gpu where the work will be run in')
    parser.add_argument('--train_batch_size', default=16, type=int,
                        help='The batch size for training')
    parser.add_argument('--val_batch_size', default=16, type=int,
                        help='The batch size for validating')
    parser.add_argument('--cache_dir', default="./cache/",
                        help='The directory to store all caches that we generate.')
    parser.add_argument('--use_cache', action='store_true', default=False,
                        help='Uses a cache for the data')
    parser.add_argument('--use_gpu', default=False, action="store_true",
                        help='The directory to store all caches that we generate.')
    parser.add_argument('--fp16', default=False, action="store_true",
                        help='Use mixed precision training.')
    parser.add_argument('--pad_len', default=128, type=int,
                        help="The amount of padding that will be added or the length that we will cut the sequences to.")
    parser.add_argument('--gradient_acc_batches', default=None, type=int,
                        help="The amount batches that we accumulate the gradient to")
    parser.add_argument('--toy_n', default=None, type=int,
                        help="N amount of examples will be used in training.")
    parser.add_argument('--num_workers', default=0, type=int,
                        help="N amount of workers for dataloading")
    parser.add_argument('-lr', '--learning_rate', default=2e-5, type=float,
                        help="N amount of workers for dataloading")
    parser.add_argument('--inject_concat', default=False, action='store_true')

    parser.add_argument('--path_to_pretrained', default=None, type=str)
    parser.add_argument('--seed',default=42,type=int)
    args = parser.parse_args()
    seed_everything(seed=args.seed)

    data_dir = args.data_dir
    model_name = args.model_save_name
    model_type = args.model_type
    model_dir = args.model_dir
    save_iters = args.save_iters
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    cache_dir = args.cache_dir
    use_gpu = args.use_gpu
    epochs = args.epochs
    pad_len = args.pad_len
    fp16 = 16 if args.fp16 else 32
    num_workers = args.num_workers
    grad_acc_batchs = args.gradient_acc_batches
    toy_n = args.toy_n
    use_cache = args.use_cache
    path_to_pretrained = args.path_to_pretrained
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    model_map = {
        "LitOutputAdapterBertModel": LitOutputAdapterBertModel,
        "LitOutputBertModel": LitOutputBertModel,
        "LitBertModel": LitBertModel,
        "LitInputBertModel": LitInputBertModel
    }
    datasets_and_models = [json.load(open(args.config_file))]

    print("Training the following", datasets_and_models)
    if model_type is not None:
        datasets_and_models = [x for x in datasets_and_models if x[3] == model_type]
        print("Selected a specifc model:", model_type)
        print(datasets_and_models)

    for idx, tup in enumerate(datasets_and_models):

        ds = tup[0]
        model = model_map[tup[1]]
        train_p = tup[2]
        model_name = tup[3]
        if model.col_fn is not None:
            ds['col_fn'] = model.col_fn

        lrm_callback = LearningRateMonitor()
        # checkpoint_callback = ModelCheckpoint(monitor="v_MAP",mode="max",dirpath="models/"+model_name+"/")
        callbacks = [lrm_callback]

        print("Starting...")
        dataset = AskUbuntuDataModule(**ds)
        if path_to_pretrained is None:
            print("Training from scratch")
            model = model(name=model_name, lr=args.learning_rate, total_steps=len(dataset.train_dataloader()) * epochs,
                          concat=args.inject_concat, **train_p)
        else:
            print("Reloading model! Applicable to loading a huggingface pytorch model.")
            model = model(name=model_name, lr=args.learning_rate, total_steps=len(dataset.train_dataloader()) * epochs,
                          concat=args.inject_concat, **train_p)
            checkpoint = torch.load(path_to_pretrained, map_location="cpu")
            model.load_state_dict(checkpoint)

        # Additional trainer params
        accumulate_grad_batches = args.gradient_acc_batches if args.gradient_acc_batches is not None else None

        print("Training", model_name)
        print("Initializing the trainer")
        exp = wandb.init(project="knowledgeinjection", name=model_name)
        wandb.watch(model, log="all")
        logger = WandbLogger(name=model_name, project="knowledgeinjection", experiment=exp)
        trainer = pl.Trainer(callbacks=callbacks, gpus=args.gpus if args.use_gpu else None,
                             auto_select_gpus=args.use_gpu, max_epochs=epochs, val_check_interval=0.25,
                             logger=logger,
                             precision=fp16, log_every_n_steps=10)
        print("Fitting...")

        trainer.fit(model, datamodule=dataset)
        print("Testing...")
        trainer.test(datamodule=dataset)
        print("Done!")
        logger.finalize("success")
        # os.makedirs("models/"+model_name+"/",exist_ok=True)
        # trainer.save_checkpoint("models/"+model_name+"/"+model_name+"_trained")
