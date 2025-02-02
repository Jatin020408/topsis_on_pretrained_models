
# Conversational AI Model Comparison Using TOPSIS

This project evaluates and ranks various Conversational AI models using the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method. The models are assessed based on multiple performance metrics, including model size, load time, inference time, max sequence length, and vocabulary size.

## 1. Models Evaluated

The following pre-trained models were tested:

* facebook/blenderbot-400M-distill
* microsoft/DialoGPT-medium
* gpt2
* EleutherAI/gpt-neo-125M
* microsoft/DialoGPT-small

## 2. Evaluation Metrics

Each model was evaluated on the following metrics:

1. Model Size (millions of parameters) – Lower is better (cost criterion)
2. Load Time (seconds) – Lower is better (cost criterion)
3. Inference Time (seconds) – Lower is better (cost criterion)
4. Max Sequence Length – Higher is better (benefit criterion)
5. Vocabulary Size – Higher is better (benefit criterion)

## 3. Methodology

1. Data Collection: The models were loaded and evaluated using the transformers library.

2. Normalization: The decision matrix was normalized to ensure fair comparison.

3. Weighting: Equal weights (0.2 each) were assigned to all criteria.

4. TOPSIS Calculation: The models were ranked based on their closeness to the ideal best and worst solutions.

5. Visualization: Bar charts and heatmaps were generated to illustrate model performance.

## 4. Results

The final ranking of models based on TOPSIS scores is stored in:

* CSV file: model_comparison_results.csv

* Graphical Output: topsis_results.png

## 5. Installation & Usage

### Prerequisites

Ensure you have the following Python libraries installed:

```
pip install numpy pandas matplotlib seaborn datasets torch transformers
```
## 6. Output

The script will print the TOPSIS scores and display the best-ranked model:




```bash
  TOPSIS Results for Conversational Models:
                                  TOPSIS Score
microsoft/DialoGPT-small                0.7317
gpt2                                    0.6431
EleutherAI/gpt-neo-125M                 0.6014
microsoft/DialoGPT-medium               0.5272
facebook/blenderbot-400M-distill        0.1813
```
Best model based on TOPSIS analysis: microsoft/DialoGPT-small

## Author

This project was developed for an AI/ML research project on conversational models. If you have any questions, feel free to reach out!

## License

MIT License
    


