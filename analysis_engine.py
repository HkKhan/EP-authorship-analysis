"""
Analysis engine for hyperprolific author study
Implements the statistical analysis and classification methods described in the paper
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import logging

from config import *
from data_models import (
    Author, Publication, AuthorType, AnalysisResults, 
    GeographicDistribution, ProductivityMetrics
)

class HyperprolificAnalysisEngine:
    """
    Main analysis engine that implements the methodology from the paper:
    1. Author classification (HA, AHA, EP)
    2. Geographic distribution analysis
    3. H-index and citation metrics
    4. Authorship position analysis
    5. Temporal trends analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def classify_authors(self, authors: List[Author]) -> Tuple[List[Author], List[Author], List[Author]]:
        """
        Classify authors as HA, AHA, or EP based on their productivity
        Returns: (HA authors, AHA authors, EP authors)
        """
        ha_authors = []
        aha_authors = []
        ep_authors = []
        
        for author in authors:
            # Check if author achieves HA/AHA status in any year
            is_ha_any_year = False
            is_aha_any_year = False
            
            for year in range(STUDY_START_YEAR, STUDY_END_YEAR + 1):
                classification = author.classify_productivity(year)
                
                if classification == AuthorType.HYPERPROLIFIC:
                    is_ha_any_year = True
                elif classification == AuthorType.ALMOST_HYPERPROLIFIC:
                    is_aha_any_year = True
            
            # Classify based on highest status achieved
            if is_ha_any_year:
                ha_authors.append(author)
                ep_authors.append(author)
            elif is_aha_any_year:
                aha_authors.append(author)
                ep_authors.append(author)
        
        self.logger.info(f"Classification complete: {len(ha_authors)} HA, {len(aha_authors)} AHA, {len(ep_authors)} EP")
        return ha_authors, aha_authors, ep_authors
    
    def analyze_geographic_distribution(self, ep_authors: List[Author]) -> GeographicDistribution:
        """
        Analyze geographic distribution of EP authors
        Maps countries to regions as defined in the paper
        """
        distribution = GeographicDistribution()
        
        for author in ep_authors:
            country = author.country
            
            # Map country to region
            region_found = False
            for region, countries in GEOGRAPHIC_REGIONS.items():
                if any(c.lower() in country.lower() for c in countries):
                    if region == "Europe":
                        distribution.europe += 1
                    elif region == "Asia":
                        distribution.asia += 1
                    elif region == "Americas":
                        distribution.americas += 1
                    elif region == "Oceania":
                        distribution.oceania += 1
                    elif region == "Africa":
                        distribution.africa += 1
                    region_found = True
                    break
            
            if not region_found:
                self.logger.warning(f"Could not map country '{country}' to a region")
        
        return distribution
    
    def calculate_productivity_metrics(self, authors: List[Author], metric_type: str = "h_index") -> ProductivityMetrics:
        """
        Calculate statistical metrics for author productivity (h-index or citations)
        """
        if metric_type == "h_index":
            values = [author.h_index for author in authors]
        elif metric_type == "citations":
            values = [author.total_citations for author in authors]
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        if not values:
            return ProductivityMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        values_series = pd.Series(values)
        
        return ProductivityMetrics(
            mean=values_series.mean(),
            median=values_series.median(),
            q1=values_series.quantile(0.25),
            q3=values_series.quantile(0.75),
            std_dev=values_series.std(),
            min_val=values_series.min(),
            max_val=values_series.max(),
            count=len(values)
        )
    
    def analyze_authorship_positions(self, authors: List[Author]) -> Dict[str, float]:
        """
        Analyze authorship position statistics for a group of authors
        Returns median percentages for first, last, and other positions
        """
        all_percentages = {
            "first": [],
            "last": [],
            "middle": []
        }
        
        for author in authors:
            percentages = author.get_authorship_percentages()
            all_percentages["first"].append(percentages["first"])
            all_percentages["last"].append(percentages["last"])
            all_percentages["middle"].append(percentages["middle"])
        
        # Calculate median percentages
        return {
            "first_median": np.median(all_percentages["first"]) if all_percentages["first"] else 0.0,
            "last_median": np.median(all_percentages["last"]) if all_percentages["last"] else 0.0,
            "middle_median": np.median(all_percentages["middle"]) if all_percentages["middle"] else 0.0,
            "first_iqr": [np.percentile(all_percentages["first"], 25), np.percentile(all_percentages["first"], 75)] if all_percentages["first"] else [0.0, 0.0],
            "last_iqr": [np.percentile(all_percentages["last"], 25), np.percentile(all_percentages["last"], 75)] if all_percentages["last"] else [0.0, 0.0]
        }
    
    def analyze_temporal_trends(self, authors: List[Author], publications: List[Publication]) -> Dict[str, Dict[int, int]]:
        """
        Analyze temporal trends in publications and EP author counts
        """
        # Annual publication counts
        annual_pub_counts = Counter(pub.year for pub in publications)
        
        # Annual EP author counts
        annual_ep_counts = {}
        
        for year in range(STUDY_START_YEAR, STUDY_END_YEAR + 1):
            ep_count = 0
            ha_count = 0
            aha_count = 0
            
            for author in authors:
                classification = author.classify_productivity(year)
                if classification == AuthorType.HYPERPROLIFIC:
                    ha_count += 1
                    ep_count += 1
                elif classification == AuthorType.ALMOST_HYPERPROLIFIC:
                    aha_count += 1
                    ep_count += 1
            
            annual_ep_counts[year] = {
                "total_ep": ep_count,
                "ha": ha_count,
                "aha": aha_count
            }
        
        return {
            "publications": dict(annual_pub_counts),
            "ep_authors": annual_ep_counts
        }
    
    def identify_most_productive_authors(self, authors: List[Author], top_n: int = 10) -> List[Dict]:
        """
        Identify and rank the most productive authors by total publications
        """
        author_productivity = []
        
        for author in authors:
            total_pubs = len(author.publications)
            max_annual = max(author.get_annual_publication_count(year) 
                           for year in range(STUDY_START_YEAR, STUDY_END_YEAR + 1))
            
            # Calculate EP duration
            ep_years = sum(1 for year in range(STUDY_START_YEAR, STUDY_END_YEAR + 1)
                         if author.classify_productivity(year).is_extremely_productive)
            
            author_productivity.append({
                "name": author.name,
                "scopus_id": author.scopus_id,
                "country": author.country,
                "total_publications": total_pubs,
                "max_annual_output": max_annual,
                "h_index": author.h_index,
                "total_citations": author.total_citations,
                "ep_duration_years": ep_years,
                "classification": "HA" if any(author.classify_productivity(year) == AuthorType.HYPERPROLIFIC 
                                            for year in range(STUDY_START_YEAR, STUDY_END_YEAR + 1)) else "AHA"
            })
        
        # Sort by total publications
        author_productivity.sort(key=lambda x: x["total_publications"], reverse=True)
        
        return author_productivity[:top_n]
    
    def analyze_ep_consistency(self, ep_authors: List[Author]) -> Dict[str, any]:
        """
        Analyze consistency of EP status across years
        """
        duration_counts = Counter()
        consistent_authors = []
        
        for author in ep_authors:
            ep_years = 0
            ep_year_list = []
            
            for year in range(STUDY_START_YEAR, STUDY_END_YEAR + 1):
                if author.classify_productivity(year).is_extremely_productive:
                    ep_years += 1
                    ep_year_list.append(year)
            
            duration_counts[ep_years] += 1
            
            # Track authors who maintained status for all 5 years
            if ep_years == 5:
                consistent_authors.append({
                    "name": author.name,
                    "total_publications": len(author.publications),
                    "classification": "HA" if any(author.classify_productivity(year) == AuthorType.HYPERPROLIFIC 
                                                for year in range(STUDY_START_YEAR, STUDY_END_YEAR + 1)) else "AHA"
                })
        
        return {
            "duration_distribution": dict(duration_counts),
            "median_duration": np.median(list(duration_counts.elements())),
            "consistent_5_year_authors": consistent_authors,
            "one_year_only_count": duration_counts[1],
            "percentage_one_year_only": (duration_counts[1] / len(ep_authors)) * 100 if ep_authors else 0
        }
    
    def run_complete_analysis(self, publications: List[Publication], authors: List[Author]) -> AnalysisResults:
        """
        Run the complete analysis pipeline following the paper's methodology
        """
        self.logger.info("Starting complete analysis...")
        
        # Step 1: Classify authors
        ha_authors, aha_authors, ep_authors = self.classify_authors(authors)
        
        # Step 2: Geographic distribution
        geo_distribution = self.analyze_geographic_distribution(ep_authors)
        
        # Step 3: Calculate productivity metrics
        h_index_metrics = {
            "HA": self.calculate_productivity_metrics(ha_authors, "h_index"),
            "AHA": self.calculate_productivity_metrics(aha_authors, "h_index"),
            "EP": self.calculate_productivity_metrics(ep_authors, "h_index")
        }
        
        citation_metrics = {
            "HA": self.calculate_productivity_metrics(ha_authors, "citations"),
            "AHA": self.calculate_productivity_metrics(aha_authors, "citations"), 
            "EP": self.calculate_productivity_metrics(ep_authors, "citations")
        }
        
        # Step 4: Temporal trends
        temporal_data = self.analyze_temporal_trends(authors, publications)
        
        # Create results object
        results = AnalysisResults(
            total_articles=len(publications),
            total_unique_authors=len(authors),
            ep_authors=ep_authors,
            ha_authors=ha_authors,
            aha_authors=aha_authors,
            geographic_distribution=geo_distribution,
            h_index_metrics=h_index_metrics,
            citation_metrics=citation_metrics,
            annual_publication_counts=temporal_data["publications"],
            annual_ep_counts={year: data["total_ep"] for year, data in temporal_data["ep_authors"].items()}
        )
        
        # Additional analyses
        self.logger.info("Running additional analyses...")
        
        # Authorship positions
        ha_authorship = self.analyze_authorship_positions(ha_authors)
        aha_authorship = self.analyze_authorship_positions(aha_authors)
        
        # Most productive authors
        top_productive = self.identify_most_productive_authors(ep_authors, 10)
        
        # EP consistency
        consistency_analysis = self.analyze_ep_consistency(ep_authors)
        
        # Log key findings
        self.logger.info("=== KEY FINDINGS ===")
        self.logger.info(f"Total articles (2020-2024): {len(publications)}")
        self.logger.info(f"Total unique authors: {len(authors)}")
        self.logger.info(f"EP authors: {len(ep_authors)} ({(len(ep_authors)/len(authors)*100):.2f}%)")
        self.logger.info(f"HA authors: {len(ha_authors)}")
        self.logger.info(f"AHA authors: {len(aha_authors)}")
        
        self.logger.info(f"Geographic distribution: Europe {geo_distribution.europe} ({geo_distribution.europe/geo_distribution.total*100:.1f}%), "
                        f"Asia {geo_distribution.asia} ({geo_distribution.asia/geo_distribution.total*100:.1f}%), "
                        f"Americas {geo_distribution.americas} ({geo_distribution.americas/geo_distribution.total*100:.1f}%)")
        
        self.logger.info(f"Peak EP year: {max(results.annual_ep_counts, key=results.annual_ep_counts.get)} "
                        f"with {max(results.annual_ep_counts.values())} EP authors")
        
        self.logger.info(f"HA authors - H-index median: {h_index_metrics['HA'].median:.0f}, "
                        f"Citations median: {citation_metrics['HA'].median:.0f}")
        
        self.logger.info(f"HA authorship positions - First: {ha_authorship['first_median']:.1f}% median, "
                        f"Last: {ha_authorship['last_median']:.1f}% median")
        
        if top_productive:
            self.logger.info(f"Most productive author: {top_productive[0]['name']} "
                           f"({top_productive[0]['total_publications']} publications)")
        
        self.logger.info(f"EP consistency - Median duration: {consistency_analysis['median_duration']:.1f} years, "
                        f"One year only: {consistency_analysis['percentage_one_year_only']:.1f}%")
        
        return results

def generate_summary_report(results: AnalysisResults) -> str:
    """
    Generate a summary report matching the format of the paper's results
    """
    summary = results.get_summary_statistics()
    
    report = f"""
HYPERPROLIFIC AUTHOR ANALYSIS RESULTS (2020-2024)

OVERVIEW:
- Total articles analyzed: {summary['total_articles_2020_2024']:,}
- Total unique authors: {summary['total_unique_authors']:,}
- Extremely Productive (EP) authors: {summary['ep_authors_count']} ({summary['ep_percentage']:.2f}%)
- Hyperprolific (HA) authors: {summary['ha_authors_count']}
- Almost Hyperprolific (AHA) authors: {summary['aha_authors_count']}

GEOGRAPHIC DISTRIBUTION:
- Europe: {summary['geographic_distribution']['europe']:.1f}%
- Asia: {summary['geographic_distribution']['asia']:.1f}%
- Americas: {summary['geographic_distribution']['americas']:.1f}%
- Oceania: {summary['geographic_distribution']['oceania']:.1f}%
- Africa: {summary['geographic_distribution']['africa']:.1f}%

PRODUCTIVITY METRICS:
HA Authors (n={results.h_index_metrics['HA'].count}):
- H-index: Mean {results.h_index_metrics['HA'].mean:.2f}, Median {results.h_index_metrics['HA'].median:.0f}
- H-index Q1-Q3: {results.h_index_metrics['HA'].q1:.0f}-{results.h_index_metrics['HA'].q3:.0f}
- Citations: Mean {results.citation_metrics['HA'].mean:.0f}, Median {results.citation_metrics['HA'].median:.0f}

AHA Authors (n={results.h_index_metrics['AHA'].count}):
- H-index: Mean {results.h_index_metrics['AHA'].mean:.2f}, Median {results.h_index_metrics['AHA'].median:.0f}
- H-index Q1-Q3: {results.h_index_metrics['AHA'].q1:.0f}-{results.h_index_metrics['AHA'].q3:.0f}
- Citations: Mean {results.citation_metrics['AHA'].mean:.0f}, Median {results.citation_metrics['AHA'].median:.0f}

TEMPORAL TRENDS:
- Peak EP year: {summary['peak_ep_year']} with {results.annual_ep_counts[summary['peak_ep_year']]} EP authors
- Median EP duration: {summary['median_ep_duration_years']:.1f} years

ANNUAL EP AUTHOR COUNTS:
{chr(10).join(f"- {year}: {count} EP authors" for year, count in results.annual_ep_counts.items())}
"""
    
    return report

if __name__ == "__main__":
    # Test with sample data
    from scopus_data_extractor import create_sample_data
    
    print("Running analysis on sample data...")
    publications, authors = create_sample_data()
    
    # Run analysis
    engine = HyperprolificAnalysisEngine()
    results = engine.run_complete_analysis(publications, authors)
    
    # Generate report
    report = generate_summary_report(results)
    print(report) 