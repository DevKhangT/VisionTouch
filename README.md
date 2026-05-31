## VisionTouch Roadmap

VisionTouch is being developed as an increasingly complex ML system, starting from a simple linear regression baseline and expanding toward a real-time multimodal AI interface.

| Stage | Focus | Short Feature / Concept List |
|---|---|---|
| 1 | Linear gaze baseline | Linear regression, iris features, face center, head pose, pixel error |
| 2 | Feature engineering + ablation | Relative iris features, normalized face features, interaction terms, ablation tests |
| 3 | Classical ML models | Ridge, Lasso, kNN, SVM, decision trees, random forest, gradient boosting |
| 4 | Temporal smoothing | Moving average, exponential smoothing, Kalman filtering, jitter reduction |
| 5 | Sequential prediction | Frame sequences, sliding windows, RNN, GRU, LSTM, TCN |
| 6 | Gesture classification | Hand landmarks, pinch/click detection, gesture labels, confusion matrix, F1 score |
| 7 | Multitask learning | Shared model, coordinate regression, gesture classification, combined losses |
| 8 | Landmark neural network | MLP, backpropagation, dropout, regularization, nonlinear landmark mapping |
| 9 | CNN vision model | Eye crops, face crops, hand crops, convolution, transfer learning |
| 10 | Vision Transformer | Image patches, patch embeddings, self-attention, ViT-style gaze prediction |
| 11 | Temporal Transformer | Attention over frame sequences, positional encoding, temporal context |
| 12 | Multimodal Transformer | Gaze + head pose + hand landmarks, fusion, cross-attention |
| 13 | Self-supervised learning | Contrastive learning, masked modeling, pretraining, representation learning |
| 14 | Reinforcement learning | Cursor control, reward design, click policy, interaction optimization |
| 15 | Uncertainty estimation | Confidence scores, prediction intervals, probabilistic regression, safe clicking |
| 16 | Multi-user generalization | Cross-user testing, domain shift, personalization, few-shot calibration |
| 17 | Privacy + edge AI | On-device inference, compression, quantization, distillation, privacy-aware ML |
| 18 | Full deployed system | Real-time interface, HCI testing, latency, usability, complete VisionTouch demo |

## Current Progress

- **Stage 1 completed:** Linear regression baseline using iris, face-center, and head-pose features.
- **Stage 2 completed:** Synthesized features and ablation testing.
- **Stage 3 next:** Classical ML model comparison using the best feature sets from Stage 2.