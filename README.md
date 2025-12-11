# Customer Support Ticket Classification

An intelligent system for automatically classifying customer support queries into appropriate departments using transformer-based deep learning.

## 📋 Overview

This project implements an automated ticket routing system that categorizes customer support inquiries into five departments:
- **Technical Support** - System issues, integration problems, software bugs
- **Billing and Payments** - Invoice queries, payment disputes, refund requests
- **Customer Service** - General inquiries, account questions, feature requests
- **Returns and Exchanges** - Product returns, exchange requests
- **Service Outages and Maintenance** - System downtime, scheduled maintenance

## 🎯 Features

- **High Accuracy**: Achieves 75.7% accuracy with weighted F1-score of 0.75
- **Fast Classification**: Real-time prediction of support ticket categories
- **User-Friendly Interface**: Gradio-based web UI for easy interaction
- **Production-Ready**: Fine-tuned DistilBERT model optimized for deployment

## 🏗️ Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Multi-class text classification (5 classes)
- **Training Data**: 9,104 customer support tickets after preprocessing
- **Train/Test Split**: 80/20

### Performance Metrics

| Department | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Billing and Payments | 0.81 | 0.85 | 0.83 | 262 |
| Customer Service | 0.67 | 0.68 | 0.67 | 499 |
| Returns and Exchanges | 0.68 | 0.46 | 0.55 | 141 |
| Service Outages | 0.77 | 0.75 | 0.76 | 113 |
| Technical Support | 0.80 | 0.83 | 0.81 | 806 |

**Overall Accuracy**: 75.7%

## 🚀 Getting Started

### Prerequisites

```bash
pip install transformers torch pandas scikit-learn gradio datasets
```

### Quick Start

1. **Clone the repository**
```bash
git clone <https://github.com/InspectorGoole/Customer-Ticket-Classification.git>
cd customer-support-classification
```

2. **Run the Jupyter notebook** (for training)
```bash
jupyter notebook final_version.ipynb
```

3. **Launch the Gradio interface** (for inference)
```bash
jupyter notebook final_gradio.ipynb
```

## 📊 Dataset

The dataset contains 10,810 customer support tickets with the following features:
- `subject`: Email subject line
- `body`: Full text of the customer query
- `answer`: Support team response
- `type`: Query type (Incident, Request, Problem, Change)
- `queue`: Department assignment (target variable)
- `priority`: Urgency level (high, medium, low)

### Data Preprocessing
- Removed 1,706 rows with missing values
- Final dataset: 9,104 tickets
- No duplicate message bodies found

## 🔧 Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 3e-5
- **Batch Size**: 16
- **Epochs**: 5
- **Max Sequence Length**: 64 tokens
- **Warmup Steps**: 500
- **Weight Decay**: 0.02

## 💡 Usage Examples

### Example 1: Technical Support Query
```
Input: "My Sennheiser Momentum 4 headphones won't connect after a firmware update. 
The app shows an error. Can you help?"

Output: Technical Support
```

### Example 2: Billing Query
```
Input: "I was charged twice for the premium support plan on June 23. 
Both charges were $19.99 through PayPal. I need a refund."

Output: Billing and Payments
```

### Example 3: Service Outage
```
Input: "The marketing agency has encountered multiple service disruptions. 
Access to critical data analytics is unavailable. This is significantly hindering our operations."

Output: Service Outages and Maintenance
```

## 📁 Project Structure

```
.
├── final_version.ipynb          # Main training notebook
├── final_bert_model/            # Saved model weights
├── final_bert_tokenizer/        # Saved tokenizer
├── customer_class.csv           # Training data
└── README.md                    # This file
```

## 🎓 Model Insights

**Strengths**:
- Excellent performance on Billing and Technical Support queries (F1 > 0.80)
- Handles diverse language styles (casual to formal)
- Robust to spelling errors and informal language

**Areas for Improvement**:
- Returns and Exchanges category shows lower recall (0.46)
- Could benefit from more training data for underrepresented classes
- Fine-tuning on domain-specific vocabulary might improve accuracy

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Adding more training data for underrepresented categories
- Implementing confidence scores in predictions
- Adding multi-label classification support
- Creating a REST API wrapper


## 👥 Authors

[Ahmed Munir Chowdhury]

## 🙏 Acknowledgments

- Built with Hugging Face Transformers
- UI powered by Gradio
- Dataset from huggingface

---

**Note**: This is a machine learning model and may occasionally misclassify tickets. Always review critical classifications before routing.
