train_autoregressive.py: Pretraining autoregressive concept structure for AR baseline (autoregressive CBM) <br>
train_concept_encoder.py: For sequential & independent training, first stage is training of concept encoder <br>

train_target_predictor.py:

- If sequential & independent training: second stage is training of target predictor
- If joint training: training of both concept encoder and target predictor

Train SCBM: Only run train_target_predictor.py <br>
