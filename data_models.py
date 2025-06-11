"""
Data models for the Hyperprolific Author Analysis.

This module defines the data structures used throughout the analysis,
including Author, Publication, and AnalysisResults classes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

@dataclass
class Author:
    """
    Represents an author with their bibliometric data.
    
    Based on the data collected from Scopus API and processed
    according to the paper methodology.
    """
    # Core identification
    scopus_id: str
    name: str
    affiliation: Optional[str] = None
    country: Optional[str] = None
    orcid: Optional[str] = None
    
    # Bibliometric metrics
    h_index: Optional[int] = None
    total_citations: Optional[int] = None
    document_count: Optional[int] = None
    
    # Publication analysis
    publications: List['Publication'] = field(default_factory=list)
    papers_per_year: Dict[int, int] = field(default_factory=dict)
    avg_papers_per_year: Optional[float] = None
    
    # Classification (from analysis)
    category: Optional[str] = None  # "HA", "AHA", or "Regular"
    is_extremely_productive: bool = False
    
    # Authorship patterns
    first_author_percentage: Optional[float] = None
    last_author_percentage: Optional[float] = None
    other_author_percentage: Optional[float] = None
    
    # Temporal data
    ep_years: List[int] = field(default_factory=list)  # Years classified as EP
    ep_duration: Optional[int] = None  # Number of EP years
    first_ep_year: Optional[int] = None
    last_ep_year: Optional[int] = None
    
    # Geographic info
    continent: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.publications:
            self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate derived metrics from publications."""
        if not self.publications:
            return
        
        # Count papers by year
        year_counts = {}
        first_author_count = 0
        last_author_count = 0
        
        for pub in self.publications:
            year = pub.year
            if year:
                year_counts[year] = year_counts.get(year, 0) + 1
                
                # Count authorship positions
                if pub.author_position == 1:
                    first_author_count += 1
                elif pub.author_position == pub.total_authors:
                    last_author_count += 1
        
        self.papers_per_year = year_counts
        
        # Calculate average papers per year
        if year_counts:
            total_papers = sum(year_counts.values())
            years_active = len(year_counts)
            self.avg_papers_per_year = total_papers / years_active if years_active > 0 else 0
        
        # Calculate authorship percentages
        total_pubs = len(self.publications)
        if total_pubs > 0:
            self.first_author_percentage = (first_author_count / total_pubs) * 100
            self.last_author_percentage = (last_author_count / total_pubs) * 100
            self.other_author_percentage = 100 - self.first_author_percentage - self.last_author_percentage
    
    def get_papers_in_year(self, year: int) -> int:
        """Get number of papers published in a specific year."""
        return self.papers_per_year.get(year, 0)
    
    def is_ep_in_year(self, year: int, threshold: int = 61) -> bool:
        """Check if author was extremely productive in a specific year."""
        return self.get_papers_in_year(year) >= threshold
    
    def calculate_ep_years(self, threshold: int = 61) -> List[int]:
        """Calculate which years the author was extremely productive."""
        ep_years = []
        for year, count in self.papers_per_year.items():
            if count >= threshold:
                ep_years.append(year)
        
        self.ep_years = sorted(ep_years)
        if self.ep_years:
            self.ep_duration = len(self.ep_years)
            self.first_ep_year = min(self.ep_years)
            self.last_ep_year = max(self.ep_years)
        
        return self.ep_years
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert author to dictionary for JSON serialization."""
        return {
            'scopus_id': self.scopus_id,
            'name': self.name,
            'affiliation': self.affiliation,
            'country': self.country,
            'continent': self.continent,
            'h_index': self.h_index,
            'total_citations': self.total_citations,
            'document_count': self.document_count,
            'avg_papers_per_year': self.avg_papers_per_year,
            'category': self.category,
            'is_extremely_productive': self.is_extremely_productive,
            'first_author_percentage': self.first_author_percentage,
            'last_author_percentage': self.last_author_percentage,
            'other_author_percentage': self.other_author_percentage,
            'ep_years': self.ep_years,
            'ep_duration': self.ep_duration,
            'papers_per_year': self.papers_per_year,
            'publication_count': len(self.publications)
        }

@dataclass
class Publication:
    """
    Represents a single publication from the analysis.
    
    Contains bibliometric data and authorship information
    extracted from Scopus.
    """
    # Core identification
    scopus_id: str
    title: str
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    
    # Citation data
    citation_count: Optional[int] = None
    
    # Author information for this publication
    authors: List[str] = field(default_factory=list)
    author_position: Optional[int] = None  # Position of target author
    total_authors: Optional[int] = None
    
    # Journal/publication info
    issn: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    
    # Subject classification
    subject_areas: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Document type
    document_type: Optional[str] = None
    
    def is_first_author(self) -> bool:
        """Check if target author is first author."""
        return self.author_position == 1
    
    def is_last_author(self) -> bool:
        """Check if target author is last author."""
        return (self.author_position is not None and 
                self.total_authors is not None and 
                self.author_position == self.total_authors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert publication to dictionary."""
        return {
            'scopus_id': self.scopus_id,
            'title': self.title,
            'year': self.year,
            'journal': self.journal,
            'citation_count': self.citation_count,
            'author_position': self.author_position,
            'total_authors': self.total_authors,
            'is_first_author': self.is_first_author(),
            'is_last_author': self.is_last_author(),
            'document_type': self.document_type
        }

@dataclass
class GeographicDistribution:
    """Geographic distribution analysis results."""
    europe: int = 0
    asia: int = 0
    americas: int = 0
    oceania: int = 0
    africa: int = 0
    other: int = 0
    
    def get_total(self) -> int:
        """Get total author count."""
        return (self.europe + self.asia + self.americas + 
                self.oceania + self.africa + self.other)
    
    def get_percentages(self) -> Dict[str, float]:
        """Get percentage distribution."""
        total = self.get_total()
        if total == 0:
            return {region: 0.0 for region in self.get_regions()}
        
        return {
            'Europe': (self.europe / total) * 100,
            'Asia': (self.asia / total) * 100,
            'Americas': (self.americas / total) * 100,
            'Oceania': (self.oceania / total) * 100,
            'Africa': (self.africa / total) * 100,
            'Other': (self.other / total) * 100
        }
    
    def get_regions(self) -> List[str]:
        """Get list of region names."""
        return ['Europe', 'Asia', 'Americas', 'Oceania', 'Africa', 'Other']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'counts': {
                'Europe': self.europe,
                'Asia': self.asia,
                'Americas': self.americas,
                'Oceania': self.oceania,
                'Africa': self.africa,
                'Other': self.other
            },
            'percentages': self.get_percentages(),
            'total': self.get_total()
        }

@dataclass
class AuthorshipPatterns:
    """Analysis of authorship position patterns."""
    first_author_median: float = 0.0
    first_author_mean: float = 0.0
    last_author_median: float = 0.0
    last_author_mean: float = 0.0
    other_author_median: float = 0.0
    other_author_mean: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'first_author_median': self.first_author_median,
            'first_author_mean': self.first_author_mean,
            'last_author_median': self.last_author_median,
            'last_author_mean': self.last_author_mean,
            'other_author_median': self.other_author_median,
            'other_author_mean': self.other_author_mean
        }

@dataclass
class ProductivityMetrics:
    """Bibliometric productivity metrics for author groups."""
    h_index_median: float = 0.0
    h_index_mean: float = 0.0
    citations_median: float = 0.0
    citations_mean: float = 0.0
    papers_per_year_median: float = 0.0
    papers_per_year_mean: float = 0.0
    
    authorship_patterns: AuthorshipPatterns = field(default_factory=AuthorshipPatterns)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'h_index_median': self.h_index_median,
            'h_index_mean': self.h_index_mean,
            'citations_median': self.citations_median,
            'citations_mean': self.citations_mean,
            'papers_per_year_median': self.papers_per_year_median,
            'papers_per_year_mean': self.papers_per_year_mean,
            'authorship_patterns': self.authorship_patterns.to_dict()
        }

@dataclass
class TemporalTrends:
    """Temporal analysis of EP author trends."""
    annual_ep_counts: Dict[int, int] = field(default_factory=dict)
    annual_ha_counts: Dict[int, int] = field(default_factory=dict)
    annual_aha_counts: Dict[int, int] = field(default_factory=dict)
    
    peak_year: Optional[int] = None
    peak_year_count: Optional[int] = None
    
    ep_duration_distribution: Dict[int, int] = field(default_factory=dict)
    median_ep_duration: Optional[float] = None
    
    def calculate_peak_year(self):
        """Calculate the peak year for EP authors."""
        if self.annual_ep_counts:
            self.peak_year = max(self.annual_ep_counts.keys(), 
                               key=lambda k: self.annual_ep_counts[k])
            self.peak_year_count = self.annual_ep_counts[self.peak_year]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'annual_ep_counts': self.annual_ep_counts,
            'annual_ha_counts': self.annual_ha_counts,
            'annual_aha_counts': self.annual_aha_counts,
            'peak_year': self.peak_year,
            'peak_year_count': self.peak_year_count,
            'ep_duration_distribution': self.ep_duration_distribution,
            'median_ep_duration': self.median_ep_duration
        }

@dataclass
class AnalysisResults:
    """
    Complete analysis results containing all findings.
    
    This is the main data structure that contains all analysis
    outputs, statistics, and visualizations data.
    """
    # Basic statistics
    total_unique_authors: int = 0
    total_publications: int = 0
    
    # Author classification counts
    ha_authors: List[Author] = field(default_factory=list)
    aha_authors: List[Author] = field(default_factory=list)
    ep_authors: List[Author] = field(default_factory=list)  # HA + AHA
    regular_authors: List[Author] = field(default_factory=list)
    
    # Key metrics
    ep_count: int = 0
    ha_count: int = 0
    aha_count: int = 0
    ep_percentage: float = 0.0
    
    # Geographic analysis
    geographic_distribution: GeographicDistribution = field(default_factory=GeographicDistribution)
    
    # Productivity metrics by group
    ha_metrics: ProductivityMetrics = field(default_factory=ProductivityMetrics)
    aha_metrics: ProductivityMetrics = field(default_factory=ProductivityMetrics)
    
    # Temporal analysis
    temporal_trends: TemporalTrends = field(default_factory=TemporalTrends)
    
    # Top productive authors
    top_productive_authors: List[Tuple[str, float]] = field(default_factory=list)
    
    # Analysis metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    study_period: Tuple[int, int] = (2020, 2024)
    journals_analyzed: int = 20
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics from author lists."""
        self.ha_count = len(self.ha_authors)
        self.aha_count = len(self.aha_authors)
        self.ep_count = self.ha_count + self.aha_count
        
        if self.total_unique_authors > 0:
            self.ep_percentage = (self.ep_count / self.total_unique_authors) * 100
    
    def get_summary_text(self) -> str:
        """Generate human-readable summary of results."""
        summary = f"""
Hyperprolific Author Analysis Results
=====================================

Study Period: {self.study_period[0]}-{self.study_period[1]}
Analysis Date: {self.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}

Overall Statistics:
- Total Unique Authors: {self.total_unique_authors:,}
- Total Publications: {self.total_publications:,}
- Journals Analyzed: {self.journals_analyzed}

Author Classification:
- Extremely Productive (EP) Authors: {self.ep_count} ({self.ep_percentage:.2f}%)
- Hyperprolific (HA) Authors: {self.ha_count}
- Almost Hyperprolific (AHA) Authors: {self.aha_count}

Geographic Distribution:
"""
        
        geo_percentages = self.geographic_distribution.get_percentages()
        for region, percentage in geo_percentages.items():
            count = getattr(self.geographic_distribution, region.lower(), 0)
            summary += f"- {region}: {count} authors ({percentage:.1f}%)\n"
        
        summary += f"""
Temporal Trends:
- Peak Year: {self.temporal_trends.peak_year} ({self.temporal_trends.peak_year_count} EP authors)
- Median EP Duration: {self.temporal_trends.median_ep_duration} years

Top 5 Most Productive Authors:
"""
        
        for i, (name, papers) in enumerate(self.top_productive_authors[:5], 1):
            summary += f"{i}. {name}: {papers:.1f} papers/year\n"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete results to dictionary for JSON export."""
        return {
            'metadata': {
                'analysis_date': self.analysis_date.isoformat(),
                'study_period': self.study_period,
                'journals_analyzed': self.journals_analyzed
            },
            'summary_statistics': {
                'total_unique_authors': self.total_unique_authors,
                'total_publications': self.total_publications,
                'ep_count': self.ep_count,
                'ha_count': self.ha_count,
                'aha_count': self.aha_count,
                'ep_percentage': self.ep_percentage
            },
            'geographic_distribution': self.geographic_distribution.to_dict(),
            'ha_metrics': self.ha_metrics.to_dict(),
            'aha_metrics': self.aha_metrics.to_dict(),
            'temporal_trends': self.temporal_trends.to_dict(),
            'top_productive_authors': self.top_productive_authors
        }
    
    def save_to_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'AnalysisResults':
        """Load results from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create instance and populate from data
        # This is a simplified loader - in practice, you'd want more robust deserialization
        results = cls()
        results.total_unique_authors = data['summary_statistics']['total_unique_authors']
        results.total_publications = data['summary_statistics']['total_publications']
        results.ep_count = data['summary_statistics']['ep_count']
        results.ha_count = data['summary_statistics']['ha_count']
        results.aha_count = data['summary_statistics']['aha_count']
        results.ep_percentage = data['summary_statistics']['ep_percentage']
        
        return results

# =============================================================================
# Utility Functions for Data Processing
# =============================================================================

def create_sample_author(scopus_id: str, name: str, papers_per_year: float, 
                        country: str = "Unknown", h_index: int = 50, 
                        citations: int = 10000) -> Author:
    """Create a sample author for testing purposes."""
    from .config import get_continent, get_author_category
    
    author = Author(
        scopus_id=scopus_id,
        name=name,
        country=country,
        continent=get_continent(country),
        h_index=h_index,
        total_citations=citations,
        avg_papers_per_year=papers_per_year,
        category=get_author_category(papers_per_year),
        is_extremely_productive=papers_per_year >= 61
    )
    
    return author

def create_sample_publication(scopus_id: str, title: str, year: int, 
                            journal: str, author_position: int = 1, 
                            total_authors: int = 5) -> Publication:
    """Create a sample publication for testing purposes."""
    return Publication(
        scopus_id=scopus_id,
        title=title,
        year=year,
        journal=journal,
        author_position=author_position,
        total_authors=total_authors,
        citation_count=0,
        document_type="article"
    )

if __name__ == "__main__":
    # Test the data models
    print("Testing data models...")
    
    # Create sample author
    author = create_sample_author("12345", "Test Author", 75.0, "Germany")
    print(f"Created author: {author.name} ({author.category})")
    
    # Create sample publication
    pub = create_sample_publication("67890", "Test Paper", 2023, "Test Journal")
    print(f"Created publication: {pub.title}")
    
    # Create analysis results
    results = AnalysisResults()
    results.total_unique_authors = 1000
    results.ha_count = 10
    results.aha_count = 15
    results.calculate_summary_statistics()
    
    print(f"EP percentage: {results.ep_percentage:.2f}%")
    print("Data models test completed successfully!") 