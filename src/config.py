"""Central configuration for the confidence-gated abstention experiments.

Every value the paper reports as a setting lives here, so a run can be
reproduced from this file alone. Paths are resolved against the repository
root, so scripts can be launched from any working directory.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------- corpus
# The IPL ecosystem, at the scale the revision guide calls for: every current
# franchise, the defunct ones, the prominent players, a spread of seasons, the
# records page, the main venues, and the surrounding cricket context. The
# domain is still narrow enough to hand-label answerability, but it is no
# longer so thin that a plausible IPL question falls outside it by accident --
# which is what makes the near-miss unanswerable queries a fair test.
#
# Titles that do not exist, or that redirect onto a page already collected,
# are reported and skipped by scrape.py rather than silently dropped.
IPL_PAGES = [
    # -- the league and its governance
    "Indian Premier League",
    "Board of Control for Cricket in India",
    "Twenty20",
    "Women's Premier League (cricket)",

    # -- current franchises
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Rajasthan Royals",
    "Gujarat Titans",
    "Lucknow Super Giants",

    # -- defunct franchises
    "Deccan Chargers",
    "Pune Warriors India",
    "Kochi Tuskers Kerala",
    "Gujarat Lions",
    "Rising Pune Supergiant",

    # -- players
    "Virat Kohli",
    "MS Dhoni",
    "Rohit Sharma",
    "Jasprit Bumrah",
    "Chris Gayle",
    "David Warner",
    "AB de Villiers",
    "Rashid Khan",
    "Hardik Pandya",
    "KL Rahul",
    "Ravindra Jadeja",
    "Suresh Raina",
    "Sunil Narine",
    "Andre Russell",
    "Yuzvendra Chahal",
    "Shikhar Dhawan",
    "Rishabh Pant",
    "Sanju Samson",
    "Shubman Gill",
    "Jos Buttler",
    "Ravichandran Ashwin",
    "Bhuvneshwar Kumar",

    # -- seasons
    "2008 Indian Premier League",
    "2016 Indian Premier League",
    "2020 Indian Premier League",
    "2023 Indian Premier League",
    "2024 Indian Premier League",
    "2025 Indian Premier League",

    # -- records and statistics
    "List of Indian Premier League records and statistics",
    "List of Indian Premier League seasons and results",
    "List of Indian Premier League hat-tricks",
    "List of Indian Premier League centuries",

    # -- venues
    "Wankhede Stadium",
    "M. Chinnaswamy Stadium",
    "Eden Gardens",
    "Narendra Modi Stadium",
    "Arun Jaitley Stadium",
    "M. A. Chidambaram Stadium",
    "Rajiv Gandhi International Cricket Stadium",
    "Sawai Mansingh Stadium",

    # -- surrounding cricket context
    "India national cricket team",
    "ICC Men's T20 World Cup",
    "Cricket World Cup",
    "Indian Cricket League",
]

# Section headings that carry no prose worth retrieving.
SKIP_SECTIONS = {
    "see also", "references", "external links", "notes", "footnotes",
    "further reading", "bibliography", "sources", "citations",
}

MIN_SECTION_CHARS = 100          # sections shorter than this are dropped
# A 200-character chunk is one or two sentences, which routinely split a
# fact across two chunks. 500 characters holds three to five sentences, so
# a retrieved passage usually carries the whole fact plus its context.
CHUNK_SIZE = 500                 # characters
CHUNK_OVERLAP = 100              # characters

# ------------------------------------------------------------- retrieval
EMBED_MODEL = "all-mpnet-base-v2"
TOP_K = 3
# faiss.IndexFlatIP over L2-normalised vectors, i.e. inner product == cosine
# similarity. This is NOT an L2-distance index; no distance-to-similarity
# transform is applied or needed.
SIMILARITY = "cosine (inner product over normalised vectors)"

# ------------------------------------------------------------- abstention
# Swept wide enough to bracket BOTH edges of the operating region. The seven
# values originally reported (0.45-0.70) all sit above the out-of-domain score
# ceiling, so they only ever show precision decaying on the answerable side;
# the low end is needed to find where abstention recall actually breaks.
THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80,
]

# The subset reported in the paper's original sweep table.
PAPER_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70]
ABSTAIN_TEXT = "I don't know based on the provided context."

# Scores assigned to an abstention without consulting the judge: it did not
# answer (relevance 1) but also did not invent anything (faithfulness 5).
ABSTAIN_RELEVANCE = 1.0
ABSTAIN_FAITHFULNESS = 5.0

# An answer counts as a hallucination if the judge scores it strictly below
# this on the 0-5 faithfulness scale.
HALLUCINATION_FAITHFULNESS_CUTOFF = 3.0

# ----------------------------------------------------------------- models
GENERATOR_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0
MAX_TOKENS = 1024                # gpt-oss models spend tokens on reasoning
RANDOM_SEED = 42

# Groq free-tier ceilings observed on this account. The per-minute budget
# is what the client throttles itself against; the per-day budget is a
# separate bucket that self-throttling cannot avoid, only outlast.
TOKENS_PER_MINUTE = 8000
TOKENS_PER_DAY = 200000

# A daily-quota 429 needs hours, not seconds, so it is retried on its own
# schedule: the client sleeps for as long as the server asks and keeps
# going until this deadline. Combined with the resumable per-query cache
# in experiments.py, a run that exhausts the daily budget picks itself up
# when the budget returns instead of losing the queries still outstanding.
DAILY_LIMIT_MAX_WAIT_SECONDS = 8 * 60 * 60
DAILY_LIMIT_POLL_SECONDS = 300

# ------------------------------------------------------------------ files
RAW_DATASET_FILE = os.path.join(DATA_DIR, "raw_dataset.json")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
QUERIES_FILE = os.path.join(DATA_DIR, "queries.json")
HUMAN_RATINGS_FILE = os.path.join(DATA_DIR, "human_ratings.json")

PER_QUERY_FILE = os.path.join(RESULTS_DIR, "per_query.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "output.json")
THRESHOLD_EXPERIMENT_FILE = os.path.join(RESULTS_DIR, "threshold_experiment.json")
SELECTED_THRESHOLD_FILE = os.path.join(RESULTS_DIR, "selected_threshold.json")
RETRIEVAL_METRICS_FILE = os.path.join(RESULTS_DIR, "retrieval_metrics.json")
CATEGORY_RESULTS_FILE = os.path.join(RESULTS_DIR, "category_results.json")
JUDGE_VALIDATION_FILE = os.path.join(RESULTS_DIR, "judge_validation.json")
PAPER_NUMBERS_FILE = os.path.join(RESULTS_DIR, "paper_numbers.json")

THRESHOLD_PLOT_FILE = os.path.join(RESULTS_DIR, "threshold_plot.png")
FINAL_PLOT_FILE = os.path.join(RESULTS_DIR, "final_plot.png")
CONFIDENCE_PLOT_FILE = os.path.join(RESULTS_DIR, "confidence_plot.png")
