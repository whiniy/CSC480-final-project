# CSC480-final-project
This is an AI model used to score the risk of accounting transactions and probability of fraud.


Ideas:
- fraud risk scoring system to audit triage where we compare interpretable Bayesian model against stronger black-box models

- Bayseian Fraud Risk Scoring for Accounting Transactions: Interpretable Audit Triage vs. Complex Machine Learning Models

- build model that gives each transaction a classification in terms of fraud risk 
- model returns P(anomalous = yes | features)
probability:
   - low risk
   - medium risk
   - high risk
   - manual audit

- interpretable Naive Bayes Model
   - transaction type: invoice, reimbursement, payroll, vendor payment
   - amount bucket: small, medium, large, extreme
   - vendor status: existing vendor, new vendor
   - time: business hours, after hours, weekend
   - approval level: approved, missing approval, late approval
   - department: finance, sales, operations, misc, unknown
   - round number: y/n
   - duplicate invoice: y/n
   - unusual account: y/n

- compare against 2-3 stronger models
   - logistic regression
   - random forest
   - gradient boosting
   - bayesian network

- use:
   - precision
   - recall
   - F1 score
   - confusion matrix
   - etc

- define fake audit cost system for True Negative, False Positive, False Negative, True Positive
- compare models by expected ML metrics AND financial cost

Transaction means for each vendor:
V01: 2931.31
V02: 3114.88
V03: 3221.67
V04: 3027.42
V05: 3022.61
V06: 2827.89
V07: 3050.69
V08: 2915.50
V09: 2894.61
V10: 3297.84
V11: 3012.75
V12: 2468.88
V13: 3022.55
V14: 3222.59
V15: 3076.65
V16: 2977.92
V17: 3001.85
V18: 2957.82
V19: 2496.47
V20: 2693.81