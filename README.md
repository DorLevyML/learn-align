# learn-align

- This is the formal implementation of
"Learning to Align the Source Code to the Compiled Object Code".
- The repository includes code to build and train the introduced models and baselines,
as well as the data used in the paper.
- For more information, see "Learning to Align the Source Code to the Compiled Object Code".
- The code was tested on TensorFlow 1.3.

## Prerequisites

- Python 2.7
- TensorFlow 1.3
- numpy 1.13.1

## Run

Just run `main.py`, and set its flags according to the wanted experiment. Examples:
- In order to train, the flag `'train'` should be set to `True`. If it is set to `False` then an existing
model will be evaluated.
- In order to load an existing model, either to evaluate it or continue training it, set the flag
`'load_existing'` to `True` and `'input_dir'` to the appropriate folder in which the model is saved.
- set `'task'` to `'nat'` or `'art'` for training the model to align either natural (real world)
or artificial (randomly generated) code, respectively.
- Set `'task'` to `'tsp5'` or `'tsp10'` for training the model on the Travelling Salesman Problem for
either n=5 or n=10, respectively.
- For the convolutional grid decoder, set `'decoder'` to `'cnn'`. For the local grid decoder
set `'decoder'` to `'rnn'` and `'rnn_dec_type'` to `'local'` (the local grid decoder is not sequential.
It is set this way due to implementational reasons only).
- For the baselines based on a sequential decoder, set `'decoder'` to `'rnn'` and set `'rnn_dec_type'`
to the wanted type of rnn decoder baseline (`'ptr1'`, `'ptr2'` or `'match_lstm'`).

#### Training and evaluation

During training, products such as checkpoints, evaluation results and other files (e.g., for use
by tensorboard) are saved in the folder `'output'`, in a subfolder whose name is the time the
experiment has begun on.

When a model is loaded for evaluation (by setting the flag `'train'` to `False`), the evaluation
results are saved in its folder, in the subfolder `'evaluations'`.

## Datasets

#### Alignment

There are two datasets for the alignment task, artificial data (`'art'`) and natural, real-world data (`'nat'`).
Each dataset was divided to three parts (train, validation, test - 0.9, 0.1, 0.1, respectively),
according to the hash modulu 10 of each sample.

#### TSP

The original datasets of the TSP task, published by the authors of "Pointer Networks", should be
downloaded and put in the folder `LearnAlign/DataSets/PtrNets_datasets/`.
The relevant files are: `tsp5`, `tsp5_test`, `tsp_10_test_exact` and `tsp_10_train_exact`.

## Tensorboard

The following data can be displayed by tensorboard over time:
- loss
- accuracy
- gradient norms in different parts of the neural network

## License

This work is licensed under the MIT license (can be found in `LICENSE` file),
except for the GNU dataset files, located in `LearnAlign/DataSets/natural_projects_datasets` and are licensed
under the GNU GPLv3 license (can be found in `LearnAlign/DataSets/natural_projects_datasets/LICENSE`).

## Other works we used for this work

We used the following works for our work.
Licenses of these works are included in the file `licenses_of_other_works.txt`.
- TensorFlow- the file `net/main.py` is based on the file: `fully_connected_feed.py`
from: `tensorflow/examples/tutorials/mnist/`
of the TensorFlow Mechanics 101 tutorial,
and was modified from it.
- The GNU project- the human-written, real-world source code data ("natural data") was taken from source code of
libraries that are part of the GNU project. The libraries source code was downloaded from: http://mirror.rackdc.com/gnu/.
The libraries whose source code was taken are:
`CSSC-1.4.0, a2ps-4.14, acct-6.6.2, acm-5.1, anubis-4.2,
archimedes-2.0.1, barcode-0.99, bash-4.4-rc2, ccd2cue-0.5, cflow-1.5,
cim-5.1, combine-0.4.0, coreutils-8.9, cpio-2.12, cvs-1.11.23,
dap-3.9, dico-2.3, diction-1.11, diffutils-3.5, direvent-5.1,
ed-1.13, enscript-1.6.6, findutils-4.6.0, garpd-0.2.0, gawk-4.1.4,
gcal-4, gdbm-1.12, gengen-1.4, gengetopt-2.22, gettext-0.19.8.1,
gforth-0.7.3, global-6.5, glpk-4.60, gmp-6.1.1, gnu-pw-mgr-2.0,
gnudos-1.9, gnuit-4.9.5, grep-2.9, groff-1.22.3, gsasl-1.8.0,
gsl-2.2.1, gss-1.0.3, gvpe-2.25, gzip-1.8, hello-2.9,
idutils-cvs, indent-2.2.9, inetutils-1.9.4, jpeg-6b, jwhois-4.0,
libcdio-0.93, libextractor-1.3, libgsasl-1.8.0, libiconv-1.14, libidn-1.33,
libmicrohttpd-0.9.50, libosip2-5.0.0, libsigsegv-2.10, libtasn1-4.9, libunistring-0.9.6,
lightning-2.1.0, m4-1.4.17, macchanger-1.6.0, mailutils-2.2, make-4.2,
marst-2.7, mcsim-5.6.5, mifluz-0.26.0, mtools-4.0.18, nano-2.7.0,
nettle-3.2, patch-2.7.5, pies-1.2, plotutils-2.6, radius-1.6,
rcs-5.9.4, readline-7.0, recutils-1.7, rush-1.7, sed-4.2,
sharutils-4.14.2, shmm-1.0, spell-1.1, swbis-1.13, tar-1.29,
texinfo-6.3, trueprint-5.4, units-2.13, wdiff-1.2.2, which-2.21`.






