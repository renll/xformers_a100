# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum

import pytorch_lightning as pl
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torch import nn
from torchmetrics import Accuracy
from torchvision import transforms

from examples.microViT import VisionTransformer
from xformers.factory import xFormer, xFormerConfig
from xformers.helpers.hierarchical_configs import (
    BasicLayerConfig,
    get_hierarchical_configuration,
)


class Classifier(str, Enum):
    GAP = "gap"
    TOKEN = "token"


class MetaVisionTransformer(VisionTransformer):
    def __init__(
        self,
        steps,
        learning_rate=5e-4,
        betas=(0.9, 0.99),
        weight_decay=0.03,
        image_size=32,
        num_classes=10,
        patch_size=2,
        dim=384,
        n_layer=6,
        n_head=6,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        mlp_pdrop=0.0,
        attention="scaled_dot_product",
        layer_norm_style="pre",
        hidden_layer_multiplier=4,
        use_rotary_embeddings=True,
        linear_warmup_ratio=0.1,
        classifier: Classifier = Classifier.TOKEN,
    ):

        super(VisionTransformer, self).__init__()

        # all the inputs are saved under self.hparams (hyperparams)
        self.save_hyperparameters()

        assert image_size % patch_size == 0

        # Generate the skeleton of our hierarchical Transformer

        # This is the small metaformer configuration,
        # truncated of the last part since the pictures are too small with CIFAR10 (32x32)
        # Any other related config would work,
        # and the attention mechanisms don't have to be the same across layers
        base_hierarchical_configs = [
            BasicLayerConfig(
                embedding=64,
                attention_mechanism=attention,
                patch_size=7,
                stride=4,
                padding=2,
                seq_len=image_size * image_size // 16,
            ),
            BasicLayerConfig(
                embedding=128,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 64,
            ),
            BasicLayerConfig(
                embedding=320,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 256,
            ),
            # BasicLayerConfig(
            #     embedding=512,
            #     attention_mechanism=attention,
            #     patch_size=3,
            #     stride=2,
            #     padding=1,
            #     seq_len=image_size * image_size // 1024,
            # ),
        ]

        # Fill in the gaps in the config
        xformer_config = get_hierarchical_configuration(
            base_hierarchical_configs,
            layernorm_style=layer_norm_style,
            use_rotary_embeddings=use_rotary_embeddings,
            mlp_multiplier=4,
            dim_head=32,
        )

        # Now instantiate the metaformer trunk
        config = xFormerConfig(xformer_config)
        print(config)
        self.trunk = xFormer.from_config(config)
        print(self.trunk)

        # The classifier head
        dim = base_hierarchical_configs[-1].embedding
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        x = self.trunk(x)
        x = self.ln(x)

        if self.hparams.classifier == Classifier.TOKEN:
            x = x[:, 0]  # only consider the token, we're classifying anyway
        elif self.hparams.classifier == Classifier.GAP:
            x = x.mean(dim=1)  # mean over sequence len

        x = self.head(x)
        return x


if __name__ == "__main__":
    pl.seed_everything(42)

    # Adjust batch depending on the available memory on your machine.
    # You can also use reversible layers to save memory
    REF_BATCH = 512
    BATCH = 256

    MAX_EPOCHS = 50
    NUM_WORKERS = 4
    GPUS = 1

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    # We'll use a datamodule here, which already handles dataset/dataloader/sampler
    # See https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html
    # for a full tutorial
    dm = CIFAR10DataModule(
        data_dir="data",
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    dm.train_transforms = train_transforms
    dm.test_transforms = test_transforms
    dm.val_transforms = test_transforms

    image_size = dm.size(-1)  # 32 for CIFAR
    num_classes = dm.num_classes  # 10 for CIFAR

    # compute total number of steps
    batch_size = BATCH * GPUS
    steps = dm.num_samples // REF_BATCH * MAX_EPOCHS
    lm = MetaVisionTransformer(
        steps=steps,
        image_size=image_size,
        num_classes=num_classes,
        attention="pooling",
        layer_norm_style="pre",
        use_rotary_embeddings=True,
    )
    trainer = pl.Trainer(
        gpus=GPUS,
        max_epochs=MAX_EPOCHS,
        precision=16,
        accumulate_grad_batches=REF_BATCH // BATCH,
    )
    trainer.fit(lm, dm)

    # check the training
    trainer.test(lm, datamodule=dm)
