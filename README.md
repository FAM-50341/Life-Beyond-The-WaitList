🧠 AI-Powered Organ Transplant Eligibility Prediction
Organ transplantation is a life-saving procedure for patients with terminal organ failure. However, with increasing demand and a limited supply of organs, there's a critical need for fair, accurate, and efficient eligibility assessment. This project introduces an AI-powered system designed to support clinicians in making data-driven decisions regarding transplant suitability.

📌 Abstract

This project proposes a predictive system for organ transplant eligibility using a synthetic dataset containing 1,345,297 records with 11 clinical, demographic, and immunological features. The model uses a stacked ensemble of:
1. XGBoost
2. LightGBM
3. Decision Tree
4. Random Forest
with Logistic Regression as the meta-classifier.
This solution supports real-time, high-precision predictions, enhancing clinical decision-making processes.

🎯 Objectives
-Develop a robust machine learning model to assess transplant eligibility.
-Identify influential factors affecting transplant decisions.
-Evaluate the model using metrics like accuracy, precision, recall, and F1-score.
-Ensure interpretability for use in clinical environments (e.g., hospital transplant committees).

⚙️ Methodology
![Methodology](https://drive.google.com/file/d/1EfAyF572D9hNADq6v4vTASEH8KTsuQQH/view?usp=sharing)

🔍 Motivation
Traditional transplant eligibility assessments often involve subjective decisions and manual scoring systems, leading to inconsistencies across institutions.
To address this, our AI system:
-Leverages large-scale historical data.
-Reduces manual bias and human error.
-Promotes consistent and equitable decision-making.
-Offers potential integration with hospital IT systems for automated screening.

📈 Results

| Model           | Accuracy | Precision | Recall | F1-Score | AUC Score |
|----------------|----------|-----------|--------|----------|-----------|
| Decision Tree  | 0.9782   | 0.96      | 0.96   | 0.96     | 0.9841    |
| Random Forest  | 0.9825   | 0.97      | 0.97   | 0.97     | 0.9912    |
| XGBoost        | 0.9828   | 0.97      | 0.97   | 0.97     | 0.9913    |
| LightGBM       | 0.9836   | 0.98      | 0.98   | 0.98     | 0.9919    |
| **Stacked Model** | **0.9849** | **0.98**  | **0.98** | **0.98**  | **0.9931** |

✅ Accuracy: 98%
🧮 Macro F1 Score: 0.52
🧮 Weighted F1 Score: 0.9757
📊 Performance: Outperformed all individual models.
💡 Captured complex non-linear interactions in eligibility decisions.

🚧 Limitations & Future Work
-Imbalanced classes: Lower macro F1 due to underrepresented categories.
- Synthetic data: Requires validation on real clinical datasets.
- Interpretability challenges: Ensemble models can be complex for clinicians.
- Limited features: Key medical data like comorbidities and lab results were not included.
- Bias risk: Synthetic data may not fully eliminate latent biases.

📚 References
T. Singh et al., Machine Learning Models for Liver Transplant Eligibility, Journal of Medical Informatics, 2021.
L. Zhao et al., AI in Kidney Transplant Prediction, Computers in Biology and Medicine, 2020.
J. Martinez et al., Deep Learning in Organ Allocation, IEEE JBHI, 2019.
D. Kim, Ethical Concerns in AI Transplant Systems, AI and Ethics, 2022.
A. Rahman et al., Comparative Study of ML for Organ Transplant, HIMS 2023.

                                                                       ❤️ Life Beyond the Waitlist
                                                                    All rights reserved by Farzana Akter Mily
