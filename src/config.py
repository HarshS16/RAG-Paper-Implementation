# Configuration file for RAG experiments
# All hardcoded values should be defined here for reproducibility

# Retrieval configuration
TOP_K = 3
EMBED_MODEL = "all-mpnet-base-v2"

# Experiment thresholds for abstention
THRESHOLDS = [0.45, 0.5, 0.55, 0.6, 0.62, 0.65, 0.7]

# File paths
DATA_DIR = "data"
RESULTS_DIR = "results"
CHUNKS_FILE = "data/chunks.json"
OUTPUT_FILE = "results/output.json"
THRESHOLD_EXPERIMENT_FILE = "results/threshold_experiment.json"
THRESHOLD_PLOT_FILE = "results/threshold_plot.png"
PERFORMANCE_PLOT_FILE = "results/performance_comparison.png"
FINAL_PLOT_FILE = "results/final_plot.png"

# Evaluation configuration
OPTIMAL_THRESHOLD = 0.55
