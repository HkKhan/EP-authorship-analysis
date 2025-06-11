"""
Analysis Engine for Hyperprolific Author Analysis.

This module implements the core analysis algorithms described in the paper,
including author classification, statistical calculations, and trend analysis.
"""

import logging
import statistics
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime

import config
from data_models import (
    Author, Publication, AnalysisResults, GeographicDistribution,
    ProductivityMetrics, AuthorshipPatterns, TemporalTrends
)

# All config constants are now accessed via config.CONSTANT_NAME

# Set up logging
logger = logging.getLogger(__name__)

class AnalysisEngine:
    """
    Core analysis engine implementing the methodology from the paper.
    
    Performs author classification, statistical analysis, and generates
    comprehensive results following the exact methodology described
    in the research paper.
    """
    
    def __init__(self):
        """Initialize the analysis engine."""
        self.results = AnalysisResults()
        logger.info("AnalysisEngine initialized")
    
    def analyze_authors(self, authors: List[Author]) -> AnalysisResults:
        """
        Perform complete analysis on a list of authors.
        
        This is the main entry point that orchestrates all analysis steps:
        1. Author classification (HA/AHA/EP)
        2. Geographic distribution analysis
        3. Productivity metrics calculation
        4. Temporal trends analysis
        5. Authorship pattern analysis
        
        Args:
            authors: List of Author objects to analyze
            
        Returns:
            AnalysisResults object containing all findings
        """
        logger.info(f"Starting analysis of {len(authors)} authors")
        
        # Initialize results
        self.results = AnalysisResults()
        self.results.total_unique_authors = len(authors)
        
        # Step 1: Classify authors by productivity
        self._classify_authors(authors)
        
        # Step 2: Calculate geographic distribution
        self._analyze_geographic_distribution()
        
        # Step 3: Calculate productivity metrics
        self._calculate_productivity_metrics()
        
        # Step 4: Analyze temporal trends
        self._analyze_temporal_trends()
        
        # Step 5: Analyze authorship patterns
        self._analyze_authorship_patterns()
        
        # Step 6: Identify top productive authors
        self._identify_top_productive_authors()
        
        # Step 7: Calculate summary statistics
        self.results.calculate_summary_statistics()
        
        # Step 8: Validate results against expected ranges
        self._validate_results()
        
        logger.info(f"Analysis complete: {self.results.ep_count} EP authors identified")
        return self.results
    
    def _classify_authors(self, authors: List[Author]) -> None:
        """
        Classify authors into HA, AHA, and regular categories.
        
        Implementation follows the paper methodology:
        - HA: ≥72 papers/year (≥1 paper every 5 days)
        - AHA: 61-72 papers/year (1 paper every 6 days)
        - EP: HA + AHA combined
        
        Args:
            authors: List of Author objects to classify
        """
        logger.info("Classifying authors by productivity...")
        
        ha_authors = []
        aha_authors = []
        regular_authors = []
        
        # Estimate total publications for authors without publication data
        total_publications = 0
        
        for author in authors:
            # Calculate average papers per year if not already calculated
            if author.avg_papers_per_year is None:
                if author.publications:
                    # Calculate from actual publications
                    year_counts = defaultdict(int)
                    for pub in author.publications:
                        if pub.year and pub.year in config.YEARS:
                            year_counts[pub.year] += 1
                    
                    if year_counts:
                        author.avg_papers_per_year = sum(year_counts.values()) / len(year_counts)
                        author.papers_per_year = dict(year_counts)
                    else:
                        author.avg_papers_per_year = 0
                else:
                    # Estimate from document count if available
                    if author.document_count:
                        # Assume document count is for entire career, estimate study period portion
                        author.avg_papers_per_year = author.document_count / 10  # Rough estimate
                    else:
                        author.avg_papers_per_year = 0
            
            # Classify based on average papers per year
            category = config.get_author_category(author.avg_papers_per_year)
            author.category = category
            author.is_extremely_productive = config.is_extremely_productive(author.avg_papers_per_year)
            
            # Add to appropriate list
            if category == "HA":
                ha_authors.append(author)
            elif category == "AHA":
                aha_authors.append(author)
            else:
                regular_authors.append(author)
            
            # Count publications
            if author.publications:
                total_publications += len(author.publications)
        
        # Store classified authors
        self.results.ha_authors = ha_authors
        self.results.aha_authors = aha_authors
        self.results.ep_authors = ha_authors + aha_authors
        self.results.regular_authors = regular_authors
        self.results.total_publications = total_publications
        
        logger.info(f"Classification complete: {len(ha_authors)} HA, {len(aha_authors)} AHA, {len(regular_authors)} regular")
    
    def _analyze_geographic_distribution(self) -> None:
        """
        Analyze geographic distribution of EP authors.
        
        Maps author countries to continents following the paper's
        geographic classification system.
        """
        logger.info("Analyzing geographic distribution...")
        
        distribution = GeographicDistribution()
        
        for author in self.results.ep_authors:
            # Ensure continent is set
            if author.continent is None and author.country:
                author.continent = config.get_continent(author.country)
            
            continent = author.continent or "Other"
            
            # Count by continent
            if continent == "Europe":
                distribution.europe += 1
            elif continent == "Asia":
                distribution.asia += 1
            elif continent == "Americas":
                distribution.americas += 1
            elif continent == "Oceania":
                distribution.oceania += 1
            elif continent == "Africa":
                distribution.africa += 1
            else:
                distribution.other += 1
        
        self.results.geographic_distribution = distribution
        
        # Log distribution
        percentages = distribution.get_percentages()
        for region, percentage in percentages.items():
            count = getattr(distribution, region.lower(), 0)
            logger.info(f"{region}: {count} authors ({percentage:.1f}%)")
    
    def _calculate_productivity_metrics(self) -> None:
        """
        Calculate bibliometric productivity metrics for HA and AHA groups.
        
        Metrics include:
        - H-index statistics (median, mean)
        - Citation statistics (median, mean)
        - Papers per year statistics
        - Authorship position patterns
        """
        logger.info("Calculating productivity metrics...")
        
        # Calculate metrics for HA authors
        if self.results.ha_authors:
            self.results.ha_metrics = self._calculate_group_metrics(
                self.results.ha_authors, "HA"
            )
        
        # Calculate metrics for AHA authors
        if self.results.aha_authors:
            self.results.aha_metrics = self._calculate_group_metrics(
                self.results.aha_authors, "AHA"
            )
    
    def _calculate_group_metrics(self, authors: List[Author], group_name: str) -> ProductivityMetrics:
        """
        Calculate productivity metrics for a specific author group.
        
        Args:
            authors: List of authors in the group
            group_name: Name of the group (for logging)
            
        Returns:
            ProductivityMetrics object with calculated statistics
        """
        if not authors:
            return ProductivityMetrics()
        
        # Extract metrics
        h_indices = [a.h_index for a in authors if a.h_index is not None and a.h_index > 0]
        citations = [a.total_citations for a in authors if a.total_citations is not None and a.total_citations > 0]
        papers_per_year = [a.avg_papers_per_year for a in authors if a.avg_papers_per_year is not None and a.avg_papers_per_year > 0]
        
        # Calculate authorship patterns
        authorship_patterns = self._calculate_authorship_patterns(authors)
        
        # Calculate statistics
        metrics = ProductivityMetrics(
            h_index_median=statistics.median(h_indices) if h_indices else 0.0,
            h_index_mean=statistics.mean(h_indices) if h_indices else 0.0,
            citations_median=statistics.median(citations) if citations else 0.0,
            citations_mean=statistics.mean(citations) if citations else 0.0,
            papers_per_year_median=statistics.median(papers_per_year) if papers_per_year else 0.0,
            papers_per_year_mean=statistics.mean(papers_per_year) if papers_per_year else 0.0,
            authorship_patterns=authorship_patterns
        )
        
        logger.info(f"{group_name} metrics: H-index median={metrics.h_index_median:.1f}, "
                   f"Citations median={metrics.citations_median:.0f}")
        
        return metrics
    
    def _calculate_authorship_patterns(self, authors: List[Author]) -> AuthorshipPatterns:
        """
        Calculate authorship position patterns for a group of authors.
        
        Args:
            authors: List of authors to analyze
            
        Returns:
            AuthorshipPatterns object with statistics
        """
        first_author_percentages = []
        last_author_percentages = []
        other_author_percentages = []
        
        for author in authors:
            if (author.first_author_percentage is not None and 
                author.last_author_percentage is not None and 
                author.other_author_percentage is not None):
                
                first_author_percentages.append(author.first_author_percentage)
                last_author_percentages.append(author.last_author_percentage)
                other_author_percentages.append(author.other_author_percentage)
        
        return AuthorshipPatterns(
            first_author_median=statistics.median(first_author_percentages) if first_author_percentages else 0.0,
            first_author_mean=statistics.mean(first_author_percentages) if first_author_percentages else 0.0,
            last_author_median=statistics.median(last_author_percentages) if last_author_percentages else 0.0,
            last_author_mean=statistics.mean(last_author_percentages) if last_author_percentages else 0.0,
            other_author_median=statistics.median(other_author_percentages) if other_author_percentages else 0.0,
            other_author_mean=statistics.mean(other_author_percentages) if other_author_percentages else 0.0
        )
    
    def _analyze_temporal_trends(self) -> None:
        """
        Analyze temporal trends in EP author productivity.
        
        Analyzes:
        - Annual counts of EP, HA, and AHA authors
        - Peak productivity years
        - Duration of EP status
        - Temporal patterns
        """
        logger.info("Analyzing temporal trends...")
        
        trends = TemporalTrends()
        
        # Count EP authors by year
        annual_ep_counts = defaultdict(int)
        annual_ha_counts = defaultdict(int)
        annual_aha_counts = defaultdict(int)
        
        ep_durations = []
        
        for author in self.results.ep_authors:
            # Calculate EP years for this author
            ep_years = []
            
            for year in config.YEARS:
                papers_in_year = author.get_papers_in_year(year)
                if papers_in_year >= config.ALMOST_HYPERPROLIFIC_THRESHOLD:
                    ep_years.append(year)
                    annual_ep_counts[year] += 1
                
                if papers_in_year >= config.HYPERPROLIFIC_THRESHOLD:
                    ha_years.append(year)
            
            # Record EP duration
            if ep_years:
                author.ep_years = ep_years
                author.ep_duration = len(ep_years)
                author.first_ep_year = min(ep_years)
                author.last_ep_year = max(ep_years)
                ep_durations.append(author.ep_duration)
        
        # Store annual counts
        trends.annual_ep_counts = dict(annual_ep_counts)
        trends.annual_ha_counts = dict(annual_ha_counts)
        trends.annual_aha_counts = dict(annual_aha_counts)
        
        # Calculate peak year
        trends.calculate_peak_year()
        
        # Calculate EP duration statistics
        if ep_durations:
            trends.median_ep_duration = statistics.median(ep_durations)
            
            # Duration distribution
            duration_counter = Counter(ep_durations)
            trends.ep_duration_distribution = dict(duration_counter)
        
        self.results.temporal_trends = trends
        
        logger.info(f"Peak year: {trends.peak_year} with {trends.peak_year_count} EP authors")
        logger.info(f"Median EP duration: {trends.median_ep_duration:.1f} years")
    
    def _analyze_authorship_patterns(self) -> None:
        """
        Analyze detailed authorship position patterns across all EP authors.
        
        This analysis is already included in productivity metrics but can be
        extended here for more detailed patterns.
        """
        logger.info("Analyzing authorship patterns...")
        
        # This is primarily handled in _calculate_productivity_metrics
        # But we can add additional analysis here if needed
        
        # Calculate overall authorship statistics for EP authors
        total_first_author = 0
        total_last_author = 0
        total_publications = 0
        
        for author in self.results.ep_authors:
            if author.publications:
                for pub in author.publications:
                    total_publications += 1
                    if pub.is_first_author():
                        total_first_author += 1
                    elif pub.is_last_author():
                        total_last_author += 1
        
        if total_publications > 0:
            first_author_rate = (total_first_author / total_publications) * 100
            last_author_rate = (total_last_author / total_publications) * 100
            
            logger.info(f"Overall EP authorship: {first_author_rate:.1f}% first author, "
                       f"{last_author_rate:.1f}% last author")
    
    def _identify_top_productive_authors(self) -> None:
        """
        Identify the most productive authors based on average papers per year.
        
        Creates a ranked list of top productive authors matching the
        format presented in the paper.
        """
        logger.info("Identifying top productive authors...")
        
        # Sort EP authors by average papers per year
        sorted_authors = sorted(
            self.results.ep_authors,
            key=lambda a: a.avg_papers_per_year or 0,
            reverse=True
        )
        
        # Extract top 20 for detailed reporting
        top_authors = []
        for author in sorted_authors[:20]:
            if author.avg_papers_per_year:
                top_authors.append((author.name, author.avg_papers_per_year))
        
        self.results.top_productive_authors = top_authors
        
        # Log top 5
        for i, (name, papers) in enumerate(top_authors[:5], 1):
            logger.info(f"{i}. {name}: {papers:.1f} papers/year")
    
    def _validate_results(self) -> None:
        """
        Validate analysis results against expected ranges from the paper.
        
        Checks if key metrics fall within reasonable ranges based on
        the original study findings.
        """
        logger.info("Validating results...")
        
        validation_results = {}
        
        # Check EP author total
        ep_total = self.results.ep_count
        expected_range = config.VALID_RANGES["ep_authors_total"]
        validation_results["ep_total"] = (expected_range[0] <= ep_total <= expected_range[1])
        
        # Check HA author count
        ha_count = self.results.ha_count
        expected_range = config.VALID_RANGES["ha_authors"]
        validation_results["ha_count"] = (expected_range[0] <= ha_count <= expected_range[1])
        
        # Check AHA author count
        aha_count = self.results.aha_count
        expected_range = config.VALID_RANGES["aha_authors"]
        validation_results["aha_count"] = (expected_range[0] <= aha_count <= expected_range[1])
        
        # Check geographic distribution
        if self.results.geographic_distribution:
            percentages = self.results.geographic_distribution.get_percentages()
            
            europe_pct = percentages.get("Europe", 0)
            expected_range = config.VALID_RANGES["europe_percentage"]
            validation_results["europe_pct"] = (expected_range[0] <= europe_pct <= expected_range[1])
            
            asia_pct = percentages.get("Asia", 0)
            expected_range = config.VALID_RANGES["asia_percentage"]
            validation_results["asia_pct"] = (expected_range[0] <= asia_pct <= expected_range[1])
        
        # Log validation results
        all_valid = all(validation_results.values())
        logger.info(f"Validation {'passed' if all_valid else 'failed'}")
        
        for metric, is_valid in validation_results.items():
            if not is_valid:
                logger.warning(f"Validation failed for {metric}")
    
    def generate_detailed_statistics(self) -> Dict[str, Any]:
        """
        Generate detailed statistics for research reporting.
        
        Returns comprehensive statistics that can be used for
        publication and detailed analysis.
        
        Returns:
            Dictionary containing detailed statistical results
        """
        stats = {
            "study_metadata": {
                "analysis_date": datetime.now().isoformat(),
                "study_period": f"{config.YEARS[0]}-{config.YEARS[-1]}",
                "total_years": len(config.YEARS),
                "journals_analyzed": 20
            },
            
            "author_classification": {
                "total_unique_authors": self.results.total_unique_authors,
                "ha_authors": self.results.ha_count,
                "aha_authors": self.results.aha_count,
                "ep_authors": self.results.ep_count,
                "ep_percentage": self.results.ep_percentage,
                "regular_authors": len(self.results.regular_authors)
            },
            
            "geographic_distribution": self.results.geographic_distribution.to_dict(),
            
            "productivity_metrics": {
                "ha_metrics": self.results.ha_metrics.to_dict(),
                "aha_metrics": self.results.aha_metrics.to_dict()
            },
            
            "temporal_trends": self.results.temporal_trends.to_dict(),
            
            "top_productive_authors": self.results.top_productive_authors[:10],
            
            "publication_statistics": {
                "total_publications": self.results.total_publications,
                "avg_publications_per_author": (
                    self.results.total_publications / self.results.total_unique_authors
                    if self.results.total_unique_authors > 0 else 0
                )
            }
        }
        
        return stats
    
    def compare_with_paper_findings(self) -> Dict[str, Any]:
        """
        Compare analysis results with the original paper findings.
        
        Returns:
            Dictionary showing comparison between current results and paper findings
        """
        
        comparison = {
            "ep_authors": {
                "paper": config.PAPER_FINDINGS["total_ep_authors"],
                "current": self.results.ep_count,
                "difference": self.results.ep_count - config.PAPER_FINDINGS["total_ep_authors"]
            },
            
            "ha_authors": {
                "paper": config.PAPER_FINDINGS["ha_authors"],
                "current": self.results.ha_count,
                "difference": self.results.ha_count - config.PAPER_FINDINGS["ha_authors"]
            },
            
            "aha_authors": {
                "paper": config.PAPER_FINDINGS["aha_authors"],
                "current": self.results.aha_count,
                "difference": self.results.aha_count - config.PAPER_FINDINGS["aha_authors"]
            },
            
            "peak_year": {
                "paper": config.PAPER_FINDINGS["peak_year"],
                "current": self.results.temporal_trends.peak_year,
                "match": (self.results.temporal_trends.peak_year == config.PAPER_FINDINGS["peak_year"])
            }
        }
        
        return comparison

def analyze_publication_patterns(authors: List[Author]) -> Dict[str, Any]:
    """
    Utility function to analyze publication patterns across authors.
    
    Args:
        authors: List of authors to analyze
        
    Returns:
        Dictionary with publication pattern statistics
    """
    patterns = {
        "yearly_distribution": defaultdict(int),
        "journal_distribution": defaultdict(int),
        "collaboration_patterns": {
            "solo_papers": 0,
            "small_teams": 0,  # 2-5 authors
            "medium_teams": 0,  # 6-10 authors
            "large_teams": 0   # 11+ authors
        },
        "document_types": defaultdict(int)
    }
    
    for author in authors:
        for pub in author.publications:
            # Year distribution
            if pub.year:
                patterns["yearly_distribution"][pub.year] += 1
            
            # Journal distribution
            if pub.journal:
                patterns["journal_distribution"][pub.journal] += 1
            
            # Collaboration patterns
            if pub.total_authors:
                if pub.total_authors == 1:
                    patterns["collaboration_patterns"]["solo_papers"] += 1
                elif pub.total_authors <= 5:
                    patterns["collaboration_patterns"]["small_teams"] += 1
                elif pub.total_authors <= 10:
                    patterns["collaboration_patterns"]["medium_teams"] += 1
                else:
                    patterns["collaboration_patterns"]["large_teams"] += 1
            
            # Document types
            if pub.document_type:
                patterns["document_types"][pub.document_type] += 1
    
    # Convert defaultdicts to regular dicts
    patterns["yearly_distribution"] = dict(patterns["yearly_distribution"])
    patterns["journal_distribution"] = dict(patterns["journal_distribution"])
    patterns["document_types"] = dict(patterns["document_types"])
    
    return patterns

def calculate_collaboration_metrics(authors: List[Author]) -> Dict[str, float]:
    """
    Calculate collaboration metrics for a group of authors.
    
    Args:
        authors: List of authors to analyze
        
    Returns:
        Dictionary with collaboration metrics
    """
    total_papers = 0
    total_coauthors = 0
    solo_papers = 0
    
    for author in authors:
        for pub in author.publications:
            total_papers += 1
            
            if pub.total_authors:
                if pub.total_authors == 1:
                    solo_papers += 1
                else:
                    total_coauthors += pub.total_authors - 1  # Exclude the author themselves
    
    metrics = {
        "average_coauthors_per_paper": total_coauthors / total_papers if total_papers > 0 else 0,
        "solo_paper_percentage": (solo_papers / total_papers * 100) if total_papers > 0 else 0,
        "collaborative_paper_percentage": ((total_papers - solo_papers) / total_papers * 100) if total_papers > 0 else 0
    }
    
    return metrics

if __name__ == "__main__":
    # Test the analysis engine
    print("Testing AnalysisEngine...")
    
    # Create sample authors for testing
    from data_models import create_sample_author
    
    test_authors = [
        create_sample_author("1", "Test HA Author", 80.0, "Germany", 70, 25000),
        create_sample_author("2", "Test AHA Author", 65.0, "Japan", 60, 15000),
        create_sample_author("3", "Test Regular Author", 20.0, "USA", 30, 5000),
    ]
    
    # Run analysis
    engine = AnalysisEngine()
    results = engine.analyze_authors(test_authors)
    
    print(f"Classified {results.ep_count} EP authors from {results.total_unique_authors} total")
    print(f"Geographic distribution: {results.geographic_distribution.to_dict()}")
    print("AnalysisEngine test completed successfully!") 