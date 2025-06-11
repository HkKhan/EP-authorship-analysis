"""
Data models for the hyperprolific author analysis
Based on the methodology described in the paper
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import pandas as pd

class AuthorType(Enum):
    """Classification of author productivity levels"""
    REGULAR = "regular"
    ALMOST_HYPERPROLIFIC = "aha"  # 61-72 papers/year
    HYPERPROLIFIC = "ha"  # 72+ papers/year
    
    @property
    def is_extremely_productive(self) -> bool:
        """EP authors are HA + AHA combined"""
        return self in [AuthorType.HYPERPROLIFIC, AuthorType.ALMOST_HYPERPROLIFIC]

class AuthorshipPosition(Enum):
    """Position of author in publication"""
    FIRST = "first"
    LAST = "last"
    MIDDLE = "middle"
    CORRESPONDING = "corresponding"

@dataclass
class Publication:
    """Represents a single publication from Scopus"""
    scopus_id: str
    title: str
    year: int
    journal: str
    authors: List[str]  # List of Scopus author IDs
    keywords: List[str]
    citation_count: int
    document_type: str
    source_id: str
    
    @property
    def num_coauthors(self) -> int:
        return len(self.authors)

@dataclass
class Author:
    """Represents an author with their publication history"""
    scopus_id: str
    name: str
    affiliation: str
    country: str
    h_index: int
    total_citations: int
    publications: List[Publication] = field(default_factory=list)
    
    def get_publications_by_year(self, year: int) -> List[Publication]:
        """Get publications for a specific year"""
        return [pub for pub in self.publications if pub.year == year]
    
    def get_annual_publication_count(self, year: int) -> int:
        """Get number of publications in a specific year"""
        return len(self.get_publications_by_year(year))
    
    def classify_productivity(self, year: int) -> AuthorType:
        """Classify author productivity for a given year"""
        annual_count = self.get_annual_publication_count(year)
        
        if annual_count >= 72:
            return AuthorType.HYPERPROLIFIC
        elif annual_count >= 61:
            return AuthorType.ALMOST_HYPERPROLIFIC
        else:
            return AuthorType.REGULAR
    
    def get_authorship_positions(self) -> Dict[AuthorshipPosition, int]:
        """Calculate authorship position statistics"""
        positions = {pos: 0 for pos in AuthorshipPosition}
        
        for pub in self.publications:
            author_index = pub.authors.index(self.scopus_id) if self.scopus_id in pub.authors else -1
            
            if author_index == 0:
                positions[AuthorshipPosition.FIRST] += 1
            elif author_index == len(pub.authors) - 1:
                positions[AuthorshipPosition.LAST] += 1
            else:
                positions[AuthorshipPosition.MIDDLE] += 1
        
        return positions
    
    def get_authorship_percentages(self) -> Dict[str, float]:
        """Get authorship position percentages"""
        positions = self.get_authorship_positions()
        total = sum(positions.values())
        
        if total == 0:
            return {"first": 0.0, "last": 0.0, "middle": 0.0}
        
        return {
            "first": (positions[AuthorshipPosition.FIRST] / total) * 100,
            "last": (positions[AuthorshipPosition.LAST] / total) * 100, 
            "middle": (positions[AuthorshipPosition.MIDDLE] / total) * 100
        }

@dataclass
class GeographicDistribution:
    """Geographic distribution of authors"""
    europe: int = 0
    asia: int = 0
    americas: int = 0
    oceania: int = 0
    africa: int = 0
    
    @property
    def total(self) -> int:
        return self.europe + self.asia + self.americas + self.oceania + self.africa
    
    def get_percentages(self) -> Dict[str, float]:
        if self.total == 0:
            return {region: 0.0 for region in ["europe", "asia", "americas", "oceania", "africa"]}
        
        return {
            "europe": (self.europe / self.total) * 100,
            "asia": (self.asia / self.total) * 100,
            "americas": (self.americas / self.total) * 100,
            "oceania": (self.oceania / self.total) * 100,
            "africa": (self.africa / self.total) * 100
        }

@dataclass
class ProductivityMetrics:
    """Statistical metrics for author productivity"""
    mean: float
    median: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    std_dev: float
    min_val: float
    max_val: float
    count: int

@dataclass
class AnalysisResults:
    """Complete results of the hyperprolific author analysis"""
    total_articles: int
    total_unique_authors: int
    ep_authors: List[Author]  # Extremely productive (HA + AHA)
    ha_authors: List[Author]  # Hyperprolific
    aha_authors: List[Author]  # Almost hyperprolific
    
    geographic_distribution: GeographicDistribution
    h_index_metrics: Dict[str, ProductivityMetrics]  # HA, AHA, EP
    citation_metrics: Dict[str, ProductivityMetrics]
    
    # Annual trends
    annual_publication_counts: Dict[int, int]
    annual_ep_counts: Dict[int, int]
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """Generate summary statistics matching the paper's results"""
        return {
            "total_articles_2020_2024": self.total_articles,
            "total_unique_authors": self.total_unique_authors,
            "ep_authors_count": len(self.ep_authors),
            "ep_percentage": (len(self.ep_authors) / self.total_unique_authors) * 100,
            "ha_authors_count": len(self.ha_authors),
            "aha_authors_count": len(self.aha_authors),
            "geographic_distribution": self.geographic_distribution.get_percentages(),
            "peak_ep_year": max(self.annual_ep_counts, key=self.annual_ep_counts.get),
            "median_ep_duration_years": self._calculate_median_ep_duration()
        }
    
    def _calculate_median_ep_duration(self) -> float:
        """Calculate median duration of EP status"""
        durations = []
        for author in self.ep_authors:
            ep_years = 0
            for year in range(2020, 2025):
                if author.classify_productivity(year).is_extremely_productive:
                    ep_years += 1
            durations.append(ep_years)
        
        return pd.Series(durations).median() if durations else 0.0 