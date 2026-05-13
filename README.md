# CSC480-final-project
This is an AI model used to score the risk of accounting transactions and probability of fraud.


Ideas:
- fraud risk scoring system to audit triage where we compare interpretable Bayesian model against stronger black-box models

- Bayseian Fraud Risk Scoring for Accounting Transactions: Interpretable Audit Triage vs. Complex Machine Learning Models

- build model that gives each transaction a classification in terms of fraud risk probability:
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