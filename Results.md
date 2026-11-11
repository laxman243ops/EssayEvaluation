Automated Essay Evaluation System
Model Performance Report  |  SBERT + LightGBM  |  Butcha Laxmana Rao
1. Overview
This report summarizes the evaluation results of an automated essay scoring model. The model combines SBERT sentence embeddings with handcrafted linguistic features (word count, lexical diversity, grammar error rate) and predicts essay scores using LightGBM. It is served through a Flask web application.
Dataset: [dataset name, number of essays, score range 0-5]
Train/test split: [e.g. 80/20]   Training rounds: 250 (see learning curve)
2. Performance Summary
<img width="707" height="241" alt="image" src="https://github.com/user-attachments/assets/89605d10-05cd-4a52-80cb-5f94806a7400" />

3. Key Observations
•	No overfitting: train and test results are very close (QWK 0.722 vs 0.712, Pearson 0.787 vs 0.769, Band Accuracy 97.27% vs 97.23%), so the model generalizes to unseen essays.
•	Stable training: the MSE learning curve drops sharply in the first ~50 rounds and flattens by ~100, with test MSE settling near 0.44.
•	High band accuracy: about 97% of predictions fall within one score band of the human rater. In the confusion matrix, 4,809 of 4,946 essays are either exact matches or off by one band.
•	Moderate exact agreement: 2,913 of 4,946 essays (about 59%) match the human score exactly. Most errors are one band away, mainly between scores 1, 2 and 3.
•	Feature importance: word count and lexical diversity contribute most, followed by grammar error rate, with the semantic (SBERT) feature last.
4. Visualizations
   <img width="974" height="316" alt="image" src="https://github.com/user-attachments/assets/d62b4c07-9045-479a-999e-9f6ba44fed84" />

 
Figure 1: Learning curve (MSE), feature importance, and human vs system score confusion matrix.
5. Limitations and Future Work
•	The model tends to pull predictions toward the middle: high-scoring essays (4 and 5) are often under-predicted and low-scoring essays (0) are often over-predicted. This is partly due to fewer samples at the extremes.
•	QWK of about 0.71 shows good but not human-level agreement. Future work includes class balancing, tuning, and adding richer features or fine-tuned transformer embeddings.
•	Evaluate on an additional external dataset to confirm generalization.
