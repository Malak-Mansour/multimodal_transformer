import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from pytorch_widedeep import Trainer
from pytorch_widedeep.callbacks import EarlyStopping, LRHistory
from pytorch_widedeep.metrics import Accuracy, F1Score
from pytorch_widedeep.models import TabTransformer, Wide, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor

sys.path.append(
    os.path.abspath("/mnt/workspace/gongkaixiong/table/run_experiments")
)  # isort:skipimport pickle
from general_utils.utils import set_lr_scheduler, set_optimizer  # noqa: E402
from parsers.tabtransformer_parser import parse_args  # noqa: E402

pd.options.display.max_columns = 100

use_cuda = torch.cuda.is_available()

args = parse_args()

ROOTDIR = Path("/mnt/workspace/gongkaixiong/table")
WORKDIR = Path(os.getcwd())

PROCESSED_DATA_DIR = ROOTDIR / "/".join(["processed_data", args.bankm_dset])
RESULTS_DIR = WORKDIR / "/".join(["results", args.bankm_dset, "tabtransformer"])
if not RESULTS_DIR.is_dir():
    os.makedirs(RESULTS_DIR)

train = pd.read_pickle(PROCESSED_DATA_DIR / "bankm_train.p")
valid = pd.read_pickle(PROCESSED_DATA_DIR / "bankm_val.p")
colnames = [c.replace(".", "_") for c in train.columns]
train.columns = colnames
valid.columns = colnames

# All columns will be treated as categorical. The column with the highest
# number of categories has 308
if args.with_wide:
    cat_embed_cols = []
    for col in train.columns:
        if train[col].nunique() >= 5 and train[col].nunique() < 400 and col != "target":
            cat_embed_cols.append(col)

    wide_cols = []
    for col in train.columns:
        if train[col].nunique() < 40 and col != "target":
            wide_cols.append(col)

    prepare_wide = WidePreprocessor(wide_cols)
    X_wide_train = prepare_wide.fit_transform(train)
    X_wide_valid = prepare_wide.transform(valid)

    prepare_tab = TabPreprocessor(embed_cols=cat_embed_cols)
    X_tab_train = prepare_tab.fit_transform(train)
    X_tab_valid = prepare_tab.transform(valid)

    y_train = train.target.values
    y_valid = valid.target.values

    wide = Wide(wide_dim=np.unique(X_wide_train).shape[0])

    X_train = {"X_wide": X_wide_train, "X_tab": X_tab_train, "target": y_train}
    X_val = {"X_wide": X_wide_valid, "X_tab": X_tab_valid, "target": y_valid}

else:
    cat_embed_cols = [c for c in train.columns if c != "target"]

    prepare_tab = TabPreprocessor(embed_cols=cat_embed_cols)
    X_tab_train = prepare_tab.fit_transform(train)
    X_tab_valid = prepare_tab.transform(valid)

    y_train = train.target.values
    y_valid = valid.target.values

    wide = None

    X_train = {"X_tab": X_tab_train, "target": y_train}
    X_val = {"X_tab": X_tab_valid, "target": y_valid}

if args.mlp_hidden_dims == "same":
    mlp_hidden_dims = [
        len(cat_embed_cols) * args.input_dim,
        len(cat_embed_cols) * args.input_dim,
        (len(cat_embed_cols) * args.input_dim) // 2,
    ]
elif args.mlp_hidden_dims == "None":
    mlp_hidden_dims = None
else:
    mlp_hidden_dims = eval(args.mlp_hidden_dims)


deeptabular = TabTransformer(column_idx=prepare_tab.column_idx, cat_embed_input=prepare_tab.cat_embed_input, continuous_cols=prepare_tab.continuous_cols, n_blocks=12, input_dim=768, mlp_hidden_dims=None)

from timm.models.vision_transformer import Block
import torch.nn as nn
ckpt = torch.load("Meta-Transformer_base_patch16_encoder.pth")
encoder = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
encoder.load_state_dict(ckpt,strict=True)
deeptabular.encoder = encoder


for n, p in deeptabular.encoder.named_parameters():
        # if 'adapter' not in n:
        p.requires_grad = False
trainables = [p for p in deeptabular.parameters() if p.requires_grad]
print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in deeptabular.parameters()) / 1e6))
print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

model = WideDeep(wide=wide, deeptabular=deeptabular)

optimizers = set_optimizer(model, args)

steps_per_epoch = (X_tab_train.shape[0] // args.batch_size) + 1
lr_schedulers = set_lr_scheduler(optimizers, steps_per_epoch, args)

early_stopping = EarlyStopping(
    monitor=args.monitor,
    min_delta=args.early_stop_delta,
    patience=args.early_stop_patience,
)

trainer = Trainer(
    model,
    objective="binary",
    optimizers=optimizers,
    lr_schedulers=lr_schedulers,
    reducelronplateau_criterion=args.monitor.split("_")[-1],
    callbacks=[early_stopping, LRHistory(n_epochs=args.n_epochs)],
    metrics=[Accuracy, F1Score],
)

start = time()
trainer.fit(
    X_train=X_train,
    X_val=X_val,
    n_epochs=args.n_epochs,
    batch_size=args.batch_size,
    validation_freq=args.eval_every,
)
runtime = time() - start

if args.save_results:
    suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
    filename = "_".join(["bankm_tabtransformer", suffix]) + ".p"
    results_d = {}
    results_d["args"] = args.__dict__
    results_d["early_stopping"] = early_stopping
    results_d["trainer_history"] = trainer.history
    results_d["runtime"] = runtime
    with open(RESULTS_DIR / filename, "wb") as f:
        pickle.dump(results_d, f)
