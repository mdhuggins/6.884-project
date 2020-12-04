import os
from argparse import ArgumentParser
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(seed=42)

from data import AskUbuntuDataModule
from models.architecturemodel import LitBertArchitectureModel
from models.basemodel import LitBertModel
from models.inputmodel import LitInputBertModel
from models.outputmodel import LitOutputBertModel
from utils import CheckpointEveryNSteps
# torch.autograd.set_detect_anomaly(True)
parser = ArgumentParser(description='Train a model to do information retrieval on askubuntu dataset')

parser.add_argument('--save_iters', type=int, default=5000,
                    help='The amount of steps to save a model')
parser.add_argument('--epochs', type=int, default=5,
                    help='The amount of epochs to run during training')
parser.add_argument('--data_dir', default="./data/askubuntu/",
                    help='The root of the data directory for the askubuntu dataset')
parser.add_argument('--model_dir', default="./models/",
                    help='The directory to save trained/checkpointed models')
parser.add_argument('--model_save_name', default="test_name",
                    help='The name to save the model with')
parser.add_argument('--model_type', default=None,
                    help='The type of model that we use for evaluation')
parser.add_argument('--gpus', default=0,type=int,
                    help='The gpu where the work will be run in')
parser.add_argument('--train_batch_size', default=16, type=int,
                    help='The batch size for training')
parser.add_argument('--val_batch_size', default=16, type=int,
                    help='The batch size for validating')
parser.add_argument('--cache_dir', default="./cache/",
                    help='The directory to store all caches that we generate.')
parser.add_argument('--use_cache', action='store_true', default= False,
                    help='Uses a cache for the data')
parser.add_argument('--use_gpu', default=False, action="store_true",
                    help='The directory to store all caches that we generate.')
parser.add_argument('--fp16', default=False, action="store_true",
                    help='Use mixed precision training.')
parser.add_argument('--pad_len',default=128,type=int,
                    help="The amount of padding that will be added or the length that we will cut the sequences to.")
parser.add_argument('--gradient_acc_batches',default=1,type=int,
                    help="The amount batches that we accumulate the gradient to")
parser.add_argument('--toy_n',default=None,type=int,
                    help="N amount of examples will be used in training.")
parser.add_argument('--num_workers',default=0,type=int,
                    help="N amount of workers for dataloading")

args = parser.parse_args()

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
if cache_dir is not None:
    os.makedirs(cache_dir, exist_ok=True)


datasets_and_models = [
    ({"pad_len":pad_len,"data_dir":data_dir,"batch_size":train_batch_size,"cache_dir":cache_dir,"use_cache":use_cache,"num_workers":num_workers,"toy_n":toy_n},
           LitBertModel,{"accumulate_grad_batches":grad_acc_batchs},"basebert"),
    ({"pad_len": pad_len, "data_dir": data_dir, "batch_size": 2, "use_cache":use_cache,"cache_dir": cache_dir, "num_workers": num_workers,"toy_n": toy_n},
           LitBertArchitectureModel, {"accumulate_grad_batches": 8}, "archbert"),
    ({"pad_len":pad_len,"data_dir":data_dir,"batch_size":train_batch_size,"use_cache":use_cache,"cache_dir":cache_dir,"num_workers":num_workers,"toy_n":toy_n},
           LitInputBertModel,{"accumulate_grad_batches":grad_acc_batchs},"inbert"),
    ({"pad_len":pad_len,"data_dir":data_dir,"batch_size":train_batch_size,"use_cache":use_cache,"cache_dir":cache_dir,"num_workers":num_workers,"toy_n":toy_n},
           LitOutputBertModel, {"accumulate_grad_batches":grad_acc_batchs},"outbert"),
    # ({"pad_len":pad_len,"data_dir":data_dir,"batch_size":train_batch_size,"cache_dir":cache_dir,"num_workers":num_workers,"toy_n":toy_n},
    #        LitOutputBaseModel, {"accumulate_grad_batches":grad_acc_batchs},"outbase"),
]
if model_type is not None:
    datasets_and_models = [x for x in datasets_and_models if x[3]==model_type]
    print("Selected a specifc model:",model_type)
    print(datasets_and_models)

for idx,tup in enumerate(datasets_and_models):
    # try:
        ds = tup[0]
        model = tup[1]
        train_p = tup[2]
        model_name = tup[3]
        if model.col_fn is not None:
            ds['col_fn'] = model.col_fn
        # checkpoints
        checkpoints = CheckpointEveryNSteps(save_iters)
        # Early stopping
        # early_stopping = EarlyStopping(monitor='val_loss_step')
        tb_logger = TensorBoardLogger(save_dir="tb_logs", name=model_name)
        print("Starting...")
        dataset = AskUbuntuDataModule(**ds)
        model = model(name=model_name)
        #Additional trainer params
        accumulate_grad_batches = train_p["accumulate_grad_batches"] if "accumulate_grad_batches" in train_p.keys() else None
        # print(dataset,model)
        # cbs = [checkpoints, early_stopping]
        print("Training",model_name)
        print("Initializing the trainer")
        trainer = pl.Trainer(callbacks=[checkpoints], gpus=[args.gpus] if args.use_gpu else None,
                             auto_select_gpus=args.use_gpu,max_epochs=epochs,val_check_interval=0.25,check_val_every_n_epoch=1,
                             logger=tb_logger,precision=fp16,num_sanity_val_steps=0
                             )
        print("Fitting...")

        # trainer.tune(model)
        # Run learning rate finder
        # lr_finder = trainer.tuner.lr_find(model,datamodule=dataset,max_lr=5)
        #
        # # Results can be found in
        # # lr_finder.results
        #
        # # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        #
        # # Pick point based on plot, or get suggestion
        # new_lr = lr_finder.suggestion()
        #
        # # update hparams of the model
        # model.hparams.lr = new_lr

        trainer.fit(model,datamodule=dataset)
        print("Testing...")
        trainer.test(datamodule=dataset)
        print("Done!")
        os.makedirs("models/"+model_name+"/")
        trainer.save_checkpoint("models/"+model_name+"/"+model_name+"_trained")
    # except Exception as e:
    #     print(e)
