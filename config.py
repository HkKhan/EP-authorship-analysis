# Configuration for Hyperprolific Author Analysis in Orthopaedic Research
# Based on the methodology described in the paper

import os
from typing import List, Dict

# Study parameters
STUDY_START_YEAR = 2020
STUDY_END_YEAR = 2024

# Author classification thresholds (papers per year)
HYPERPROLIFIC_THRESHOLD = 72  # At least one paper every 5 days
ALMOST_HYPERPROLIFIC_MIN = 61  # One paper every 6 days
ALMOST_HYPERPROLIFIC_MAX = 72

# Top 20 orthopaedic journals by CiteScore (as mentioned in paper)
TOP_ORTHOPAEDIC_JOURNALS = [
    "British Journal of Sports Medicine",
    "Sports Medicine", 
    "Journal of Sport and Health Science",
    "Journal of Cachexia, Sarcopenia and Muscle",
    "Journal of Orthopaedic Translation",
    "Osteoarthritis and Cartilage",
    "Journal of Bone and Mineral Research",
    "Exercise and Sport Sciences Reviews",
    "Bone and Joint Journal",
    "American Journal of Sports Medicine",
    "Arthroscopy: Journal of Arthroscopic and Related Surgery",
    "Skeletal Muscle",
    "Journal of Bone and Joint Surgery",
    "Physical Education and Sport Pedagogy",
    "Biology of Sport",
    "Spine Journal",
    "Knee Surgery, Sports Traumatology, Arthroscopy",
    "Calcified Tissue International",
    "Scandinavian Journal of Medicine and Science in Sports",
    "Annals of Physical and Rehabilitation Medicine"
]

# Scopus document types to include
INCLUDED_DOCTYPES = ["article", "review", "article in press"]
EXCLUDED_DOCTYPES = ["editorial", "note", "letter", "correction", "conference paper"]

# Geographic regions for analysis
GEOGRAPHIC_REGIONS = {
    "Europe": ["Germany", "United Kingdom", "Spain", "Italy", "France", "Netherlands", "Switzerland"],
    "Asia": ["Japan", "China", "South Korea", "India", "Singapore", "Taiwan"],
    "Americas": ["United States", "Canada", "Brazil", "Mexico"],
    "Oceania": ["Australia", "New Zealand"],
    "Africa": ["South Africa", "Egypt", "Nigeria"]
}

# API Configuration (would be loaded from environment)
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY', 'your_api_key_here')
SCOPUS_BASE_URL = "https://api.elsevier.com/content/search/scopus"

# Cache settings
ENABLE_CACHING = True
CACHE_DIR = "cache"

# Analysis parameters
MIN_CITATIONS_FOR_QUALITY = 1
MIN_COAUTHORS_FOR_COLLABORATION = 2 