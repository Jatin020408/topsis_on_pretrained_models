import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
import time

def evaluate_model(model_name):
    """
    Evaluate a pre-trained model on various metrics
    Returns: Dict of metrics
    """
    metrics = {}
    
    try:
        # Load model and tokenizer
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        load_time = time.time() - start_time
        
        # Get model size
        model_size = sum(p.numel() for p in model.parameters()) / 1e6  # In millions
        
        # Benchmark inference speed
        input_text = "Hello, how are you doing today?"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(input_ids, max_length=50)
        inference_time = time.time() - start_time
        
        metrics = {
            'model_size': model_size,
            'load_time': load_time,
            'inference_time': inference_time,
            'max_sequence_length': model.config.max_position_embeddings,
            'vocab_size': len(tokenizer)
        }
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return None
    
    return metrics

def apply_topsis(decision_matrix, weights, criteria_type):
    """
    Apply TOPSIS method
    decision_matrix: DataFrame with models as rows and criteria as columns
    weights: List of weights for each criterion
    criteria_type: List of '+' for benefit and '-' for cost criteria
    """
    # Normalize the decision matrix
    normalized = decision_matrix.copy()
    for column in decision_matrix.columns:
        normalized[column] = decision_matrix[column] / np.sqrt((decision_matrix[column]**2).sum())
    
    # Weight the normalized matrix
    weighted = normalized * weights
    
    # Determine ideal and negative-ideal solutions
    ideal = []
    negative_ideal = []
    
    for i, column in enumerate(weighted.columns):
        if criteria_type[i] == '+':
            ideal.append(weighted[column].max())
            negative_ideal.append(weighted[column].min())
        else:
            ideal.append(weighted[column].min())
            negative_ideal.append(weighted[column].max())
    
    # Calculate separation measures
    S_plus = np.sqrt(((weighted - ideal)**2).sum(axis=1))
    S_minus = np.sqrt(((weighted - negative_ideal)**2).sum(axis=1))
    
    # Calculate relative closeness
    C = S_minus / (S_plus + S_minus)
    
    return C

def plot_results(df, scores):
    """
    Create visualizations for the results
    """
    plt.figure(figsize=(12, 6))
    
    # Bar plot of TOPSIS scores
    plt.subplot(1, 2, 1)
    plt.bar(df.index, scores)
    plt.title('TOPSIS Scores by Model')
    plt.xticks(rotation=45)
    plt.ylabel('TOPSIS Score')
    
    # Heatmap of normalized metrics
    plt.subplot(1, 2, 2)
    sns.heatmap(df, annot=True, cmap='YlOrBr')
    plt.title('Normalized Metrics Heatmap')
    plt.tight_layout()
    
    return plt

# Define models to evaluate
models = [
    'facebook/blenderbot-400M-distill',
    'microsoft/DialoGPT-medium',
    'gpt2',
    'EleutherAI/gpt-neo-125M',
    'microsoft/DialoGPT-small'
]

# Collect metrics for each model
results = {}
for model_name in models:
    metrics = evaluate_model(model_name)
    if metrics:
        results[model_name] = metrics

# Create decision matrix
df = pd.DataFrame(results).T

# Define weights and criteria type
# Weights: model_size (0.2), load_time (0.2), inference_time (0.2), 
#         max_sequence_length (0.2), vocab_size (0.2)
weights = [0.2, 0.2, 0.2, 0.2, 0.2]
criteria_type = ['-', '-', '-', '+', '+']  # - for cost, + for benefit

# Apply TOPSIS
topsis_scores = apply_topsis(df, weights, criteria_type)

# Add scores to dataframe
df['TOPSIS Score'] = topsis_scores

# Sort by TOPSIS score
df = df.sort_values('TOPSIS Score', ascending=False)

# Create visualizations
plt = plot_results(df, topsis_scores)
plt.savefig('topsis_results.png')

# Save results to CSV
df.to_csv('model_comparison_results.csv')

print("\nTOPSIS Results for Conversational Models:")
print(df[['TOPSIS Score']].round(4))
print("\nBest model based on TOPSIS analysis:", df.index[0])