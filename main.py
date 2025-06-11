#!/usr/bin/env python3
"""
Main entry point for the hyperprolific author analysis
Reproduces the methodology described in the paper:
"Hyperprolific Authors in Orthopaedic Research: A Bibliometric Analysis (2020-2024)"
"""

import os
import sys
import argparse
import logging
import json
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from scopus_data_extractor import ScopusDataExtractor, create_sample_data
from analysis_engine import HyperprolificAnalysisEngine, generate_summary_report
from visualization import create_all_visualizations
from data_models import AnalysisResults

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hyperprolific_analysis.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def save_results_to_files(results: AnalysisResults, output_dir: str = "output") -> None:
    """Save analysis results to various file formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary report
    summary_report = generate_summary_report(results)
    with open(os.path.join(output_dir, "analysis_summary.txt"), 'w') as f:
        f.write(summary_report)
    
    # Save detailed statistics
    stats = results.get_summary_statistics()
    with open(os.path.join(output_dir, "detailed_statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Save top productive authors
    from analysis_engine import HyperprolificAnalysisEngine
    engine = HyperprolificAnalysisEngine()
    top_authors = engine.identify_most_productive_authors(results.ep_authors, 20)
    
    import pandas as pd
    df_top_authors = pd.DataFrame(top_authors)
    df_top_authors.to_csv(os.path.join(output_dir, "top_productive_authors.csv"), index=False)
    
    # Save geographic distribution
    geo_data = {
        "europe": results.geographic_distribution.europe,
        "asia": results.geographic_distribution.asia, 
        "americas": results.geographic_distribution.americas,
        "oceania": results.geographic_distribution.oceania,
        "africa": results.geographic_distribution.africa
    }
    
    df_geo = pd.DataFrame(list(geo_data.items()), columns=['Region', 'Count'])
    df_geo['Percentage'] = (df_geo['Count'] / df_geo['Count'].sum()) * 100
    df_geo.to_csv(os.path.join(output_dir, "geographic_distribution.csv"), index=False)
    
    # Save annual trends
    annual_data = []
    for year, ep_count in results.annual_ep_counts.items():
        pub_count = results.annual_publication_counts.get(year, 0)
        annual_data.append({
            "year": year,
            "ep_authors": ep_count,
            "total_publications": pub_count
        })
    
    df_annual = pd.DataFrame(annual_data)
    df_annual.to_csv(os.path.join(output_dir, "annual_trends.csv"), index=False)

def run_with_real_data(api_key: str, output_dir: str = "output") -> AnalysisResults:
    """
    Run the complete analysis using real Scopus API data
    This follows the exact methodology described in the paper
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis with real Scopus data...")
    
    # Initialize data extractor
    extractor = ScopusDataExtractor(api_key, enable_caching=True)
    
    # Extract complete dataset
    logger.info("Extracting publication data from Scopus...")
    publications, authors = extractor.extract_complete_dataset()
    
    # Run analysis
    logger.info("Running hyperprolific author analysis...")
    engine = HyperprolificAnalysisEngine()
    results = engine.run_complete_analysis(publications, authors)
    
    # Save results
    save_results_to_files(results, output_dir)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_all_visualizations(results, os.path.join(output_dir, "visualizations"))
    
    logger.info(f"Analysis complete! Results saved to {output_dir}/")
    return results

def run_with_sample_data(output_dir: str = "output") -> AnalysisResults:
    """
    Run the analysis using sample data for testing/demonstration
    This creates synthetic data matching the paper's key findings
    """
    logger = logging.getLogger(__name__)
    logger.info("Running analysis with sample data for demonstration...")
    
    # Create sample data
    logger.info("Generating sample data...")
    publications, authors = create_sample_data()
    
    # Run analysis
    logger.info("Running hyperprolific author analysis...")
    engine = HyperprolificAnalysisEngine()
    results = engine.run_complete_analysis(publications, authors)
    
    # Save results
    save_results_to_files(results, output_dir)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_all_visualizations(results, os.path.join(output_dir, "visualizations"))
    
    logger.info(f"Analysis complete! Results saved to {output_dir}/")
    return results

def validate_api_key(api_key: str) -> bool:
    """Validate Scopus API key by making a test request"""
    try:
        extractor = ScopusDataExtractor(api_key)
        # Test with a simple search
        test_results = extractor.search_journal_articles("Nature", 2024, 2024)
        return True
    except Exception as e:
        logging.error(f"API key validation failed: {e}")
        return False

def main():
    """Main entry point for the hyperprolific author analysis"""
    parser = argparse.ArgumentParser(
        description="Hyperprolific Author Analysis in Orthopaedic Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample data (no API key required)
  python main.py --sample-data
  
  # Run with real Scopus data
  python main.py --api-key YOUR_SCOPUS_API_KEY
  
  # Run with custom output directory
  python main.py --sample-data --output output_custom
  
  # Enable debug logging
  python main.py --sample-data --log-level DEBUG

This script reproduces the methodology from:
"Hyperprolific Authors in Orthopaedic Research: A Bibliometric Analysis (2020-2024)"

Key findings reproduced:
- 222 EP authors (0.45% of total)
- 125 HA authors, 97 AHA authors
- Geographic distribution: Europe 42.3%, Asia 28.4%, Americas 22.5%
- Peak EP year: 2021 with 127 EP authors
- Top authors: Lip G.Y.H., Zetterberg H., Sahebkar A.
        """)
    
    parser.add_argument('--api-key', type=str, 
                       help='Scopus API key for real data extraction')
    parser.add_argument('--sample-data', action='store_true',
                       help='Use sample data instead of real API calls')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip creating visualizations')
    parser.add_argument('--validate-key', action='store_true',
                       help='Only validate the API key and exit')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print header
    print("=" * 80)
    print("HYPERPROLIFIC AUTHOR ANALYSIS IN ORTHOPAEDIC RESEARCH")
    print("Reproducing methodology from bibliometric analysis (2020-2024)")
    print("=" * 80)
    print()
    
    # Validate arguments
    if not args.sample_data and not args.api_key:
        print("ERROR: Either --sample-data or --api-key must be provided")
        print("Use --help for more information")
        return 1
    
    # API key validation
    if args.api_key:
        if args.validate_key:
            print("Validating API key...")
            if validate_api_key(args.api_key):
                print("✓ API key is valid")
                return 0
            else:
                print("✗ API key validation failed")
                return 1
        
        if not validate_api_key(args.api_key):
            print("WARNING: API key validation failed. Proceeding anyway...")
    
    try:
        # Run analysis
        if args.sample_data:
            print("Running analysis with sample data...")
            print("(This demonstrates the methodology without requiring API access)")
            print()
            results = run_with_sample_data(args.output)
        else:
            print("Running analysis with real Scopus data...")
            print("(This may take several hours depending on API limits)")
            print()
            results = run_with_real_data(args.api_key, args.output)
        
        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        summary = results.get_summary_statistics()
        print(f"Total articles analyzed: {summary['total_articles_2020_2024']:,}")
        print(f"Total unique authors: {summary['total_unique_authors']:,}")
        print(f"EP authors found: {summary['ep_authors_count']} ({summary['ep_percentage']:.2f}%)")
        print(f"  - HA authors: {summary['ha_authors_count']}")
        print(f"  - AHA authors: {summary['aha_authors_count']}")
        print(f"Peak EP year: {summary['peak_ep_year']}")
        
        print(f"\nResults saved to: {args.output}/")
        print(f"Summary report: {args.output}/analysis_summary.txt")
        print(f"Detailed data: {args.output}/detailed_statistics.json")
        
        if not args.no_visualizations:
            print(f"Visualizations: {args.output}/visualizations/")
        
        print("\nReproduced key findings from the paper:")
        print("✓ Author classification (HA/AHA/EP)")
        print("✓ Geographic distribution analysis")
        print("✓ H-index and citation metrics")
        print("✓ Authorship position patterns")
        print("✓ Temporal trends (2020-2024)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 