# English ↔ Bengali Neural Machine Translation

A bidirectional English--Bengali machine translation project built with
**PyTorch**, using a **BiLSTM encoder--decoder with Bahdanau attention**
and a shared **SentencePiece BPE tokenizer**.

The project translates in both directions:

-   English → Bengali
-   Bengali → English

The main purpose of this project was hands-on learning: to understand
and practically implement SentencePiece tokenization, dynamic padding, a
bidirectional LSTM encoder, an attention mechanism, and a
sequence-to-sequence training and inference pipeline.

------------------------------------------------------------------------

## Project Goals

This was primarily an implementation-focused learning project. The goals
were to:

-   Train and use a shared SentencePiece tokenizer with
    `model_type="bpe"`.
-   Understand how subword tokenization works for two languages with
    different scripts.
-   Implement dynamic padding so each batch is padded only to the
    longest sequence in that batch.
-   Build a BiLSTM-based encoder--decoder architecture.
-   Implement Bahdanau (additive) attention with source-padding masking.
-   Train a single model for both translation directions using language
    tags.
-   Build a simple greedy-decoding translation function.

------------------------------------------------------------------------

## Model Architecture

The model is a sequence-to-sequence network implemented in PyTorch.

  Component                Configuration
  ------------------------ ------------------------------------
  Architecture             BiLSTM Encoder--Decoder
  Attention                Bahdanau (Additive) Attention
  Embedding                Shared source and target embedding
  Embedding dimension      100
  Encoder                  1-layer bidirectional LSTM
  Encoder hidden size      128 per direction
  Encoder output size      256
  Decoder                  1-layer LSTM
  Decoder hidden size      256
  Attention context size   256
  Output vocabulary size   8,000
  Total parameters         5,899,840
  Trainable parameters     5,899,840
  Framework                PyTorch

### Translation direction

A language tag is added at the beginning of each source sequence:

-   `<2bn>` tells the model to translate into Bengali.
-   `<2en>` tells the model to translate into English.

The decoder starts with `<s>` and generates the target sequence. The
same tokenizer, embedding table, encoder--decoder model, and output
vocabulary are used for both directions.

### Attention

Bahdanau attention calculates a context vector from the encoder outputs
at each decoder step. Padded source positions are masked before the
attention softmax so that the model does not attend to padding tokens.

------------------------------------------------------------------------

## Dataset

The project uses a sampled parallel English--Bengali corpus.

  Detail                               Value
  ------------------------ -----------------
  Sampled parallel pairs             100,000
  Pairs after cleaning                98,989
  Training pairs                      79,191
  Validation pairs                     9,899
  Test pairs                           9,899
  Split                      80% / 10% / 10%
  Random seed                             42

Pairs are retained when both the English and Bengali sentences contain
between 1 and 50 words.

Each parallel pair is used in both directions. Therefore, the training
split produces 158,382 directional training examples.

------------------------------------------------------------------------

## Tokenization --- SentencePiece BPE

A single shared SentencePiece tokenizer is trained on the English and
Bengali text from the training split.

  Setting                   Value
  ------------------------- ---------------
  Tokenizer                 SentencePiece
  Model type                `bpe`
  Vocabulary size           8,000
  Character coverage        1.0
  Maximum sequence length   64 BPE tokens

Reserved token IDs:

    ID Token     Purpose
  ---- --------- ------------------------
     0 `<pad>`   Padding
     1 `<unk>`   Unknown token
     2 `<s>`     Decoder start token
     3 `</s>`    End of sequence
     4 `<2bn>`   Translate into Bengali
     5 `<2en>`   Translate into English

Using one shared vocabulary allows both languages to use the same
embedding and output layers. BPE also splits rare words into smaller
subword units.

------------------------------------------------------------------------

## Dynamic Padding

The data loader uses a custom `collate_batch()` function.

Instead of padding every sentence to the global maximum length, each
batch is padded only to the length of its longest sequence (up to the
64-token limit).

-   Batch size: `32`
-   Training loader: shuffled
-   Validation and test loaders: not shuffled
-   Padding tokens are ignored by the loss function.

This implementation helped me understand how variable-length sequences
are prepared efficiently for batched sequence modelling.

------------------------------------------------------------------------

## Training Configuration

  Setting                     Value
  --------------------------- ------------------------------------
  Optimizer                   AdamW
  Learning rate               0.001
  Weight decay                0.01
  Loss                        Cross Entropy
  Padding ignored by loss     `ignore_index=0`
  Maximum configured epochs   1,000
  Gradient clipping           1.0
  Early stopping patience     10
  Batch size                  32
  Device                      CUDA when available, otherwise CPU

Training and validation loss and accuracy are tracked and plotted.

### Training time and stopping point

The model has approximately **5.9 million trainable parameters**, with a
large output projection layer (`Linear(512 → 8000)`) accounting for a
substantial part of the parameter count.

Training was very time-consuming. Based on the observed training speed,
completing the full run was estimated to take around **30 hours**. For
this learning experiment, I manually stopped training after **65
epochs** rather than running the configured 1,000 epochs.

At the stopping point:

-   Epoch reached: **65**
-   Best validation loss recorded: **3.3371**
-   Last training loss: **3.0942**
-   Last validation loss: **3.3454**

**Important:** This is a partially trained learning experiment, not a
claim of fully converged or production-ready translation quality. The
notebook's training configuration allowed up to 1,000 epochs, but the
run shown here was manually stopped at epoch 65.

------------------------------------------------------------------------

## Sample Predictions

The notebook's `translate()` function supports both directions.

### English → Bengali

  Input                     Model output
  ------------------------- --------------------------------
  `How are you?`            `কেমন করে?`
  `I am fine.`              `আমি ভাল আছি।`
  `I am lost.`              `আমি হারিয়ে গিয়েছি`
  `Please help me.`         `প্লিজ আমাকে আমাকে সাহায্য করো।`
  `Do you speak English?`   `ইংরেজিতে ভাষা কথা বল?`

### Bengali → English

  Input                   Model output
  ----------------------- ----------------------------
  `শুভ রাত্রি`              `Shiveness night.`
  `আমি স্কুলে যাবো`         `I will school.`
  `একটু আছে বলুন`           `Take a little bit.`
  `এটার দাম কত?`          `How much is this prices?`
  `আমি তোমাকে ভালোবাসি`   `I love you.`

These are example outputs from the current checkpoint, included to show
the inference pipeline in action. They also show that translation
quality is not consistent yet: some outputs are understandable or
correct, while others contain grammatical or lexical errors. Output
quality depends on what the model has learned during training.

------------------------------------------------------------------------

## Inference

The notebook provides a translation function:

``` python
translate(text, target_lang="bn", max_len=64)
```

Examples:

``` python
translate("How are you?", target_lang="bn")
translate("আমি তোমাকে ভালোবাসি", target_lang="en")
```

-   `target_lang="bn"` translates into Bengali.
-   `target_lang="en"` translates into English.
-   Decoding uses greedy argmax selection.
-   Generation stops when `</s>` is produced or the maximum output
    length is reached.

The current implementation re-encodes the source sequence during each
decoding step. This keeps the inference logic simple for learning, but
it is less efficient than caching the encoder outputs.

------------------------------------------------------------------------

## Project Structure

```text
Machine Translation/
│
├── artifacts/
│   ├── spm_train.txt
│   ├── spm.model
│   ├── spm.vocab
│   ├── test_ids.parquet
│   ├── test.parquet
│   ├── train_ids.parquet
│   ├── train.parquet
│   ├── val_ids.parquet
│   └── val.parquet
│
├── dataset/
│   ├── data_preparation.ipynb
│   ├── final_dataset.parquet
│   ├── train-00000-of-00005.parquet
│   └── train-00001-of-00005.parquet
│
├── best_model.pt
├── last_checkpoint.pt
└── main_notebook.ipynb
```

- `artifacts/`: Contains the SentencePiece tokenizer model, vocabulary, tokenizer training text, and prepared train, validation, and test datasets with their tokenized ID versions.
- `dataset/`: Contains the data preparation notebook, final dataset, and training dataset Parquet shards.
- `best_model.pt`: Stores the best model checkpoint saved during training.
- `last_checkpoint.pt`: Stores the latest training checkpoint and can be used to resume training.
- `main_notebook.ipynb`: Main notebook containing model architecture, training, validation, evaluation, visualization, and translation inference.

------------------------------------------------------------------------

## What I Learned

Through this project, I practised:

1.  Preparing and cleaning a parallel text dataset.
2.  Training a shared SentencePiece BPE tokenizer.
3.  Working with subword token IDs and special language tokens.
4.  Implementing dynamic padding with a custom PyTorch collate function.
5.  Building a bidirectional LSTM encoder.
6.  Connecting an encoder and decoder in a Seq2Seq architecture.
7.  Implementing Bahdanau attention and masking padded source positions.
8.  Training, validating, checkpointing, and resuming a model.
9.  Plotting training and validation metrics.
10. Implementing greedy decoding for bidirectional translation.

The main achievement of this project was the practical implementation
and understanding of these components---not achieving state-of-the-art
translation performance.

------------------------------------------------------------------------

## Limitations and Possible Improvements

-   Training was stopped at epoch 65, so the model may benefit from
    further training.
-   Some generated translations are inaccurate or grammatically
    incorrect.
-   Greedy decoding can produce weaker results than more advanced
    decoding strategies.
-   Re-encoding the source at every decoder step makes inference slower.
-   Future experiments could explore beam search, cached encoder
    outputs, training optimizations, and a larger or better-curated
    parallel dataset.

------------------------------------------------------------------------

## Technologies

-   Python
-   PyTorch
-   SentencePiece
-   NumPy
-   Pandas
-   Matplotlib
-   Jupyter Notebook

------------------------------------------------------------------------

## Note

This project was developed as a practical deep-learning and NLP learning
exercise to understand how a neural machine translation system works
internally, from subword tokenization and batching to encoder--decoder
modelling, attention, training, and inference.
