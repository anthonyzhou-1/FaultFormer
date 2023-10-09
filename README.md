# fault

Transformer_base
- Larger models need more data. Larger models don't usually outperform smaller ones
- Can do well on training data (97-98% accuracy), but seems to stagnate at 50% validation accuracy.
    - Probably a sign of overfitting, since the validation loss also goes up, while the training loss tends to decrease constantly

Transformer_cls
- Did better right away, perhaps since there is less data needed to generate a classification it is easier to train
- Base model had higher training acc, but cls model had higher validation acc

Transformer_mask
- Training seems to learn something, but perhaps needs more epochs. 
- Fine tuning
    - Need to throw out decoder
        - Doesn't work well when the decoder is kept
- Pretraining objective doesn't seem to help
- Could try using a contrastive loss rather than reconstruction loss

Fourier features
- Much faster than in the spatial domain
- Signal seems to be pretty well reconstructed with top 40 modes
- Gets better accuracy than using the spatial domain (both valid and training)

Triplet loss
- t-SNE embeddings of fourier features show no correlations
- Neither does after training with triplet loss tho, sad.
    - My guess is that most negative examples are already too far
    - After a little bit there are no more samples to learn useful trends
- Probably try data augmentations

Data Augs
- Works well when directly applied to supervised learning
- Gets to 98% validation accuracy
- Kind of slow, since need to compute fft every time for randomly generated augmentations.
- Strategy
    - Regular signal 10% of the time
    - 40% choose random augmentation
    - 40% choose 2 random augmentations

Final runs
- params = {
    "batch_size": 1024,
    "epochs": 11000,
    "d_in": 3,
    "d_model": 140,
    "nhead": 20,
    "d_hid": 300,
    "nlayers": 6,
    "dropout": 0.3,
    "warmup": 4000,
    "seq_len": 40,
    "d_lin": 512,
    "d_out": 64,
    "freeze": False,
    "n_classes": 10,
    "lr_head": None,
    "model": "Transformer_cls",
    "path": None,
    "fourier": True,
    "margin": .05,
    "p_no_aug": .1,
    "p_two_aug": .5,
    "fold": 0,
}
- About 6 hours to train
- 99.358% 5-fold validation (early stopping)
- 97.894% taking last epoch
- about 1M parameters (985,870)
- AlexNet (63M), ResNet-50 (23M), Vanilla Transformer (65M)



