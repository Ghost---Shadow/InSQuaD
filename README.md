# InSQuaD: In-Context Learning for Efficient Retrieval via Submodular Mutual Information to Enforce Quality and Diversity

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![CI](https://github.com/Ghost---Shadow/quaild/actions/workflows/python_ci.yml/badge.svg)](https://github.com/Ghost---Shadow/quaild/actions/workflows/python_ci.yml)

InSQuaD is a research framework for efficient in-context learning that leverages submodular mutual information to optimize the quality-diversity tradeoff in example selection for large language models. This implementation supports various retrieval methods, subset selection strategies, and generative models for comprehensive evaluation across multiple datasets.

## 🚀 Features

- **Submodular Optimization**: Implementation of facility location and graph cut losses for quality-diversity tradeoffs
- **Multiple Retrieval Methods**: Support for semantic search models (MPNet, sentence transformers) and dense indexes (FAISS)
- **Diverse Datasets**: Pre-configured loaders for MRPC, SST, MNLI, DBPedia, RTE, HellaSwag, XSum, MultiWOZ, and GeoQ
- **Flexible Architecture**: Modular design supporting various generative models (OpenAI, HuggingFace transformers)
- **Comprehensive Evaluation**: Built-in metrics and analysis tools for experimental evaluation
- **Experiment Management**: YAML-based configuration system with Weights & Biases integration

## 📋 Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Required API keys (OpenAI, Weights & Biases)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ghost---Shadow/quaild.git
   cd quaild
   ```

2. **Install dependencies**:
   ```bash
   source ./devops/install.sh
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory with your API keys:
   ```bash
   OPENAI_API_KEY=your_openai_key_here
   WANDB_API_KEY=your_wandb_key_here
   ```

## 🚦 Quick Start

### Running Experiments

1. **Single experiment**:
   ```bash
   python src/train.py experiments/tests/quaild_test_experiment.yaml
   ```

2. **Full experiment suite**:
   ```bash
   sh run_all_experiments.sh
   ```

3. **Offline evaluation**:
   ```bash
   python src/offline_eval.py path/to/experiment/config.yaml
   ```

### Configuration

See `experiments/` directory for configuration examples.

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
# Test everything (some tests may fail on Windows)
python -m unittest discover -s src -p "*_test.py"

# Test specific modules
python -m unittest discover -s src.dataloaders -p "*_test.py"
python -m unittest discover -s src.dense_indexes -p "*_test.py"
python -m unittest discover -s src.shortlist_strategies -p "*_test.py"
python -m unittest discover -s src.subset_selection_strategies -p "*_test.py"
```

## 🔧 Development

### Code Formatting

Format code using Black:
```bash
black .
```

### Project Structure

```
src/
├── dataloaders/          # Dataset loading and preprocessing
├── dense_indexes/        # FAISS and other dense retrieval indexes  
├── generative_models/    # LLM wrappers (OpenAI, HuggingFace)
├── losses/              # Submodular loss functions
├── semantic_search_models/ # Embedding models
├── shortlist_strategies/ # Example selection strategies
├── subset_selection_strategies/ # Submodular optimization
└── training_strategies/  # Training loops and algorithms
```

## 📊 Supported Datasets

- **MRPC**: Microsoft Research Paraphrase Corpus
- **SST**: Stanford Sentiment Treebank (binary and 5-class)
- **MNLI**: Multi-Genre Natural Language Inference
- **DBPedia**: Database entity classification
- **RTE**: Recognizing Textual Entailment
- **HellaSwag**: Commonsense reasoning
- **XSum**: Extractive summarization
- **MultiWOZ**: Task-oriented dialogue
- **GeoQ**: Geographic question answering

## 🤖 Supported Models

### Generative Models
- OpenAI GPT models (GPT-3.5, GPT-4)
- HuggingFace transformers (Gemma, T5, etc.)
- Custom model implementations

### Semantic Search Models
- MPNet (all-mpnet-base-v2)
- Sentence Transformers
- Custom embedding models

## 📈 Results and Analysis

The framework includes comprehensive analysis tools:

- **Performance Tables**: Automated LaTeX table generation
- **Visualization**: Plotting utilities for results analysis  
- **Statistical Analysis**: Confidence intervals and significance tests
- **Time Analysis**: Efficiency comparisons across methods

Results are automatically logged to Weights & Biases for easy tracking and comparison.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{insquad2025,
  title={InSQuaD: In-Context Learning for Efficient Retrieval via Submodular Mutual Information to Enforce Quality and Diversity},
  author={Nanda, Souradeep and Majee, Anay and Iyer, Rishab Krishnan},
  booktitle={Proceedings of the 2025 IEEE International Conference on Data Mining (ICDM)},
  year={2025},
  organization={IEEE},
  url={https://github.com/Ghost---Shadow/quaild}
}
```

## 🆘 Support

For questions, issues, or feature requests, please open an issue on GitHub or contact the maintainers.
