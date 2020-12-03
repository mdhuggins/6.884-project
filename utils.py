import os
from typing import Dict, List

import pytorch_lightning as pl
from tqdm import tqdm


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
            self,
            save_step_frequency,
            prefix="N-Step-Checkpoint",
            use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def get_extended_attention_mask(attention_mask, input_shape, device):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.is_decoder:
        #     batch_size, seq_length = input_shape
        #     seq_ids = torch.arange(seq_length, device=device)
        #     causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        #     # causal and attention masks must have same type with pytorch version < 1.3
        #     causal_mask = causal_mask.to(attention_mask.dtype)
        #     extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        # else:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape
            )
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def transfer_batch_to_device(batch, device):
    # print("### DEVICE CHECK", device)
    for k in batch.keys():
        if isinstance(batch[k], Dict):
            batch[k] = transfer_batch_to_device(batch[k], device)
        elif isinstance(batch[k], List):
            # print(batch[k])
            continue
        else:
            batch[k] = batch[k].to(device)
    return batch


import pandas as pd


def load_vectors_pandas(path, cache="nb.h5",clean_names = False):
    numberbatch = None
    if os.path.exists(cache):
        print("Using cached!")
        numberbatch = pd.read_hdf(cache, "mat")
    else:
        names = []
        vecs = []
        with open(path) as nb:
            for line in tqdm(nb):
                line = line.strip()
                if len(line.split()) == 2:
                    continue
                name = line.split()[0]
                if clean_names:
                    name = name.lower().replace("_"," ")
                d = pd.Series([float(x) for x in line.split()[1:]])
                vecs.append(d)
                names.append(name)
        numberbatch = pd.DataFrame(data=vecs, index=names)
        numberbatch.to_hdf(cache, "mat")
    return numberbatch
