"""
Configuration settings for the Hyperprolific Author Analysis.

This module contains all the constants, thresholds, and configuration
data used in the analysis as described in the paper.
"""

import os
from typing import Dict, List, Optional

# =============================================================================
# API Configuration
# =============================================================================

# Scopus API settings
SCOPUS_API_BASE_URL = "https://api.elsevier.com/content/search/scopus"
SCOPUS_AUTHOR_API_URL = "https://api.elsevier.com/content/author"
API_RATE_LIMIT = 1.0  # Requests per second
API_TIMEOUT = 30  # Seconds

# Get API key from environment variable
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY')

# =============================================================================
# Analysis Parameters from Paper
# =============================================================================

# Study period
START_YEAR = 2020
END_YEAR = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))

# Author classification thresholds (papers per year)
HYPERPROLIFIC_THRESHOLD = 72  # ≥72 papers/year (≥1 paper every 5 days)
ALMOST_HYPERPROLIFIC_THRESHOLD = 61  # 61-72 papers/year (1 paper every 6 days)

# Document types to include
DOCUMENT_TYPES = ['ar', 'ip', 're']  # Articles, articles in press, reviews
DOCTYPE_QUERY = "DOCTYPE(ar) OR DOCTYPE(ip) OR DOCTYPE(re)"

# =============================================================================
# Top 20 Orthopaedic Journals by CiteScore (from paper)
# =============================================================================

TOP_ORTHOPAEDIC_JOURNALS = {
    # Journal name: (ISSN, CiteScore rank)
    "Nature Reviews Rheumatology": ("1759-4790", 1),
    "Bone Research": ("2095-4700", 2),
    "Journal of Bone and Mineral Research": ("0884-0431", 3),
    "Osteoarthritis and Cartilage": ("1063-4584", 4),
    "Arthritis & Rheumatism": ("0004-3591", 5),
    "Annals of the Rheumatic Diseases": ("0003-4967", 6),
    "Nature Reviews Endocrinology": ("1759-5029", 7),
    "Bone & Joint Journal": ("2049-4394", 8),
    "Journal of Orthopaedic Research": ("0736-0266", 9),
    "Arthritis Research & Therapy": ("1478-6354", 10),
    "Clinical Orthopaedics and Related Research": ("0009-921X", 11),
    "Spine": ("0362-2436", 12),
    "American Journal of Sports Medicine": ("0363-5465", 13),
    "Journal of Bone and Joint Surgery (American)": ("0021-9355", 14),
    "Knee Surgery, Sports Traumatology, Arthroscopy": ("0942-2056", 15),
    "Arthroscopy": ("0749-8063", 16),
    "European Spine Journal": ("0940-6719", 17),
    "Journal of Shoulder and Elbow Surgery": ("1058-2746", 18),
    "Rheumatology": ("1462-0324", 19),
    "Seminars in Arthritis and Rheumatism": ("0049-0172", 20)
}

# Build journal ISSN list for queries
JOURNAL_ISSNS = [issn for issn, _ in TOP_ORTHOPAEDIC_JOURNALS.values()]

# =============================================================================
# Geographic Classification
# =============================================================================

# Country to continent mapping (major countries from the paper)
COUNTRY_TO_CONTINENT = {
    # Europe (42.3% of EP authors)
    "Germany": "Europe",
    "United Kingdom": "Europe", 
    "UK": "Europe",
    "Great Britain": "Europe",
    "Spain": "Europe",
    "Italy": "Europe",
    "France": "Europe",
    "Netherlands": "Europe",
    "Switzerland": "Europe",
    "Austria": "Europe",
    "Belgium": "Europe",
    "Sweden": "Europe",
    "Norway": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Poland": "Europe",
    "Czech Republic": "Europe",
    "Hungary": "Europe",
    "Portugal": "Europe",
    "Ireland": "Europe",
    "Greece": "Europe",
    "Russia": "Europe",
    
    # Asia (28.4% of EP authors)
    "Japan": "Asia",
    "China": "Asia",
    "South Korea": "Asia",
    "India": "Asia",
    "Singapore": "Asia",
    "Hong Kong": "Asia",
    "Taiwan": "Asia",
    "Thailand": "Asia",
    "Malaysia": "Asia",
    "Indonesia": "Asia",
    "Philippines": "Asia",
    "Israel": "Asia",
    "Turkey": "Asia",
    
    # Americas (22.5% of EP authors)
    "United States": "Americas",
    "USA": "Americas",
    "US": "Americas",
    "Canada": "Americas",
    "Brazil": "Americas",
    "Mexico": "Americas",
    "Argentina": "Americas",
    "Chile": "Americas",
    "Colombia": "Americas",
    "Peru": "Americas",
    
    # Oceania (2.7% of EP authors)
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    
    # Africa (1.4% of EP authors)
    "South Africa": "Africa",
    "Egypt": "Africa",
    "Nigeria": "Africa",
    "Morocco": "Africa",
    "Tunisia": "Africa",
    "Kenya": "Africa",
    "Ghana": "Africa"
}

# Expected geographic distribution (from paper results)
EXPECTED_GEOGRAPHIC_DISTRIBUTION = {
    "Europe": 42.3,
    "Asia": 28.4,
    "Americas": 22.5,
    "Oceania": 2.7,
    "Africa": 1.4,
    "Other": 2.7
}

# =============================================================================
# Sample Data Configuration
# =============================================================================

# Key findings from the paper to reproduce in sample data
PAPER_FINDINGS = {
    "total_unique_authors": 49236,
    "total_ep_authors": 222,
    "ep_percentage": 0.45,
    "ha_authors": 125,
    "aha_authors": 97,
    "peak_year": 2021,
    "peak_year_ep_count": 127,
    
    # Top productive authors from paper
    "top_authors": [
        ("Lip, G.Y.H.", 95.2),
        ("Zetterberg, H.", 91.6),
        ("Sahebkar, A.", 88.4),
        ("Blennow, K.", 87.2),
        ("Banach, M.", 84.8),
        ("Chen, Y.", 82.6),
        ("Li, Y.", 81.4),
        ("Zhang, Y.", 80.2),
        ("Wang, Y.", 79.8),
        ("Liu, Y.", 78.6)
    ],
    
    # Productivity metrics from paper
    "ha_metrics": {
        "h_index_median": 71,
        "citations_median": 20934,
        "first_authorship_median": 2.0,
        "last_authorship_median": 27.0,
        "other_authorship_median": 60.0
    },
    
    "aha_metrics": {
        "h_index_median": 58,
        "citations_median": 11592,
        "first_authorship_median": 2.0,
        "last_authorship_median": 23.0,
        "other_authorship_median": 60.0
    },
    
    # Annual trends
    "annual_ep_counts": {
        2020: 95,
        2021: 127,
        2022: 118,
        2023: 102,
        2024: 87
    }
}

# =============================================================================
# Analysis Configuration
# =============================================================================

# Output settings
DEFAULT_OUTPUT_DIR = "output"
CACHE_DIR = "cache"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Visualization settings
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)
COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Statistical analysis settings
CONFIDENCE_INTERVAL = 0.95
RANDOM_SEED = 42

# =============================================================================
# Scopus Query Templates
# =============================================================================

# Base query for orthopaedic literature
BASE_QUERY_TEMPLATE = (
    "PUBYEAR > {start_year} AND PUBYEAR < {end_year} AND "
    "({doctype_query}) AND ({journal_query})"
)

# Journal query builder
def build_journal_query(issns: List[str]) -> str:
    """Build Scopus journal query from ISSN list."""
    return " OR ".join([f'ISSN("{issn}")' for issn in issns])

# Full query for the study
def get_study_query() -> str:
    """Get the complete Scopus query used in the study."""
    journal_query = build_journal_query(JOURNAL_ISSNS)
    return BASE_QUERY_TEMPLATE.format(
        start_year=START_YEAR - 1,  # PUBYEAR > 2019
        end_year=END_YEAR + 1,      # PUBYEAR < 2025
        doctype_query=DOCTYPE_QUERY,
        journal_query=journal_query
    )

# Author search query template
AUTHOR_QUERY_TEMPLATE = "AU-ID({author_id})"

# =============================================================================
# Validation Settings
# =============================================================================

# Expected ranges for validation
VALID_RANGES = {
    "ep_authors_total": (200, 250),
    "ha_authors": (120, 130),
    "aha_authors": (90, 100),
    "europe_percentage": (40, 45),
    "asia_percentage": (25, 30),
    "americas_percentage": (20, 25)
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_continent(country: str) -> str:
    """Map country name to continent."""
    return COUNTRY_TO_CONTINENT.get(country, "Other")

def is_hyperprolific(papers_per_year: float) -> bool:
    """Check if author meets hyperprolific threshold."""
    return papers_per_year >= HYPERPROLIFIC_THRESHOLD

def is_almost_hyperprolific(papers_per_year: float) -> bool:
    """Check if author meets almost hyperprolific threshold."""
    return (papers_per_year >= ALMOST_HYPERPROLIFIC_THRESHOLD and 
            papers_per_year < HYPERPROLIFIC_THRESHOLD)

def is_extremely_productive(papers_per_year: float) -> bool:
    """Check if author is extremely productive (HA or AHA)."""
    return papers_per_year >= ALMOST_HYPERPROLIFIC_THRESHOLD

def get_author_category(papers_per_year: float) -> str:
    """Classify author based on papers per year."""
    if is_hyperprolific(papers_per_year):
        return "HA"
    elif is_almost_hyperprolific(papers_per_year):
        return "AHA"
    else:
        return "Regular"

# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config() -> bool:
    """Validate configuration settings."""
    # Check journal count
    if len(TOP_ORTHOPAEDIC_JOURNALS) != 20:
        print(f"Warning: Expected 20 journals, found {len(TOP_ORTHOPAEDIC_JOURNALS)}")
        return False
    
    # Check thresholds
    if ALMOST_HYPERPROLIFIC_THRESHOLD >= HYPERPROLIFIC_THRESHOLD:
        print("Error: AHA threshold must be less than HA threshold")
        return False
    
    # Check year range
    if START_YEAR >= END_YEAR:
        print("Error: Start year must be before end year")
        return False
    
    print("Configuration validation passed")
    return True

if __name__ == "__main__":
    # Print configuration summary
    print("Hyperprolific Author Analysis Configuration")
    print("=" * 50)
    print(f"Study period: {START_YEAR}-{END_YEAR}")
    print(f"Journals analyzed: {len(TOP_ORTHOPAEDIC_JOURNALS)}")
    print(f"HA threshold: ≥{HYPERPROLIFIC_THRESHOLD} papers/year")
    print(f"AHA threshold: {ALMOST_HYPERPROLIFIC_THRESHOLD}-{HYPERPROLIFIC_THRESHOLD-1} papers/year")
    print(f"Expected EP authors: {PAPER_FINDINGS['total_ep_authors']}")
    print(f"API key configured: {'Yes' if SCOPUS_API_KEY else 'No'}")
    
    # Validate configuration
    validate_config() 