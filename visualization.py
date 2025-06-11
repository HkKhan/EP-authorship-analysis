"""
Visualization Module for Hyperprolific Author Analysis.

This module creates charts, graphs, and visualizations based on the analysis
results from the paper, including geographic distribution, temporal trends,
and productivity metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .data_models import AnalysisResults, Author
from .config import YEARS, JOURNALS

# Set up visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationGenerator:
    """
    Generate visualizations for hyperprolific author analysis.
    
    Creates publication-quality charts and graphs that reproduce
    the visualizations from the original research paper.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualization generator.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300
        })
    
    def create_all_visualizations(self, results: AnalysisResults) -> List[str]:
        """
        Create all standard visualizations from the analysis results.
        
        Args:
            results: AnalysisResults object containing analysis findings
            
        Returns:
            List of file paths for created visualizations
        """
        created_files = []
        
        # Create each visualization
        try:
            created_files.append(self.plot_geographic_distribution(results))
            created_files.append(self.plot_temporal_trends(results))
            created_files.append(self.plot_productivity_metrics(results))
            created_files.append(self.plot_authorship_patterns(results))
            created_files.append(self.plot_author_classification(results))
            created_files.append(self.plot_h_index_distribution(results))
            created_files.append(self.plot_collaboration_patterns(results))
            created_files.append(self.plot_annual_publication_trends(results))
            
            # Create summary dashboard
            created_files.append(self.create_summary_dashboard(results))
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        
        return [f for f in created_files if f is not None]
    
    def plot_geographic_distribution(self, results: AnalysisResults) -> Optional[str]:
        """
        Create geographic distribution visualization.
        
        Shows the continental distribution of EP authors as both
        bar chart and pie chart.
        
        Args:
            results: Analysis results containing geographic data
            
        Returns:
            File path of saved visualization
        """
        if not results.geographic_distribution:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get data
        geo_data = results.geographic_distribution.to_dict()
        continents = list(geo_data.keys())
        counts = list(geo_data.values())
        
        # Remove zero counts
        non_zero_data = [(cont, count) for cont, count in zip(continents, counts) if count > 0]
        if not non_zero_data:
            return None
        
        continents, counts = zip(*non_zero_data)
        
        # Calculate percentages
        total = sum(counts)
        percentages = [count/total * 100 for count in counts]
        
        # Bar chart
        bars = ax1.bar(continents, counts, color=sns.color_palette("husl", len(continents)))
        ax1.set_title('Geographic Distribution of EP Authors\n(Absolute Counts)', fontweight='bold')
        ax1.set_ylabel('Number of Authors')
        ax1.set_xlabel('Continent')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # Pie chart
        colors = sns.color_palette("husl", len(continents))
        wedges, texts, autotexts = ax2.pie(counts, labels=continents, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax2.set_title('Geographic Distribution of EP Authors\n(Percentages)', fontweight='bold')
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / "geographic_distribution.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(filepath)
    
    def plot_temporal_trends(self, results: AnalysisResults) -> Optional[str]:
        """
        Create temporal trends visualization.
        
        Shows the annual counts of EP, HA, and AHA authors over time.
        
        Args:
            results: Analysis results containing temporal data
            
        Returns:
            File path of saved visualization
        """
        if not results.temporal_trends or not results.temporal_trends.annual_ep_counts:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Get data
        years = sorted(results.temporal_trends.annual_ep_counts.keys())
        ep_counts = [results.temporal_trends.annual_ep_counts.get(year, 0) for year in years]
        ha_counts = [results.temporal_trends.annual_ha_counts.get(year, 0) for year in years]
        aha_counts = [results.temporal_trends.annual_aha_counts.get(year, 0) for year in years]
        
        # Plot 1: Stacked bar chart
        width = 0.6
        x = np.arange(len(years))
        
        p1 = ax1.bar(x, ha_counts, width, label='HA (≥72 papers/year)', color='#d62728')
        p2 = ax1.bar(x, aha_counts, width, bottom=ha_counts, label='AHA (61-72 papers/year)', color='#ff7f0e')
        
        ax1.set_title('Annual Count of Extremely Productive Authors', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Number of Authors')
        ax1.set_xlabel('Year')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (ha, aha) in enumerate(zip(ha_counts, aha_counts)):
            if ha > 0:
                ax1.text(i, ha/2, str(ha), ha='center', va='center', fontweight='bold', color='white')
            if aha > 0:
                ax1.text(i, ha + aha/2, str(aha), ha='center', va='center', fontweight='bold', color='white')
        
        # Plot 2: Line chart showing trends
        ax2.plot(years, ep_counts, marker='o', linewidth=3, markersize=8, label='Total EP Authors', color='#2ca02c')
        ax2.plot(years, ha_counts, marker='s', linewidth=2, markersize=6, label='HA Authors', color='#d62728')
        ax2.plot(years, aha_counts, marker='^', linewidth=2, markersize=6, label='AHA Authors', color='#ff7f0e')
        
        ax2.set_title('Temporal Trends in Author Productivity', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Number of Authors')
        ax2.set_xlabel('Year')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Highlight peak year
        if results.temporal_trends.peak_year:
            peak_year = results.temporal_trends.peak_year
            peak_count = results.temporal_trends.peak_year_count
            ax2.annotate(f'Peak: {peak_year}\n({peak_count} authors)',
                        xy=(peak_year, peak_count),
                        xytext=(peak_year, peak_count + max(ep_counts) * 0.1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / "temporal_trends.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(filepath)
    
    def plot_productivity_metrics(self, results: AnalysisResults) -> Optional[str]:
        """
        Create productivity metrics comparison visualization.
        
        Compares H-index and citation metrics between HA and AHA groups.
        
        Args:
            results: Analysis results containing productivity data
            
        Returns:
            File path of saved visualization
        """
        if not results.ha_metrics or not results.aha_metrics:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        groups = ['HA Authors', 'AHA Authors']
        
        # H-index metrics
        h_index_median = [results.ha_metrics.h_index_median, results.aha_metrics.h_index_median]
        h_index_mean = [results.ha_metrics.h_index_mean, results.aha_metrics.h_index_mean]
        
        # Citation metrics
        citations_median = [results.ha_metrics.citations_median, results.aha_metrics.citations_median]
        citations_mean = [results.ha_metrics.citations_mean, results.aha_metrics.citations_mean]
        
        # H-index comparison (bar chart)
        x = np.arange(len(groups))
        width = 0.35
        
        ax1.bar(x - width/2, h_index_median, width, label='Median', color='skyblue', alpha=0.8)
        ax1.bar(x + width/2, h_index_mean, width, label='Mean', color='orange', alpha=0.8)
        ax1.set_title('H-Index Comparison', fontweight='bold')
        ax1.set_ylabel('H-Index')
        ax1.set_xticks(x)
        ax1.set_xticklabels(groups)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (med, mean) in enumerate(zip(h_index_median, h_index_mean)):
            ax1.text(i - width/2, med + max(h_index_median) * 0.01, f'{med:.1f}', ha='center', va='bottom')
            ax1.text(i + width/2, mean + max(h_index_mean) * 0.01, f'{mean:.1f}', ha='center', va='bottom')
        
        # Citations comparison (bar chart)
        ax2.bar(x - width/2, citations_median, width, label='Median', color='lightcoral', alpha=0.8)
        ax2.bar(x + width/2, citations_mean, width, label='Mean', color='purple', alpha=0.8)
        ax2.set_title('Total Citations Comparison', fontweight='bold')
        ax2.set_ylabel('Total Citations')
        ax2.set_xticks(x)
        ax2.set_xticklabels(groups)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (med, mean) in enumerate(zip(citations_median, citations_mean)):
            ax2.text(i - width/2, med + max(citations_median) * 0.01, f'{med:.0f}', ha='center', va='bottom')
            ax2.text(i + width/2, mean + max(citations_mean) * 0.01, f'{mean:.0f}', ha='center', va='bottom')
        
        # Papers per year comparison
        papers_median = [results.ha_metrics.papers_per_year_median, results.aha_metrics.papers_per_year_median]
        papers_mean = [results.ha_metrics.papers_per_year_mean, results.aha_metrics.papers_per_year_mean]
        
        ax3.bar(x - width/2, papers_median, width, label='Median', color='lightgreen', alpha=0.8)
        ax3.bar(x + width/2, papers_mean, width, label='Mean', color='red', alpha=0.8)
        ax3.set_title('Papers per Year Comparison', fontweight='bold')
        ax3.set_ylabel('Papers per Year')
        ax3.set_xticks(x)
        ax3.set_xticklabels(groups)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (med, mean) in enumerate(zip(papers_median, papers_mean)):
            ax3.text(i - width/2, med + max(papers_median) * 0.01, f'{med:.1f}', ha='center', va='bottom')
            ax3.text(i + width/2, mean + max(papers_mean) * 0.01, f'{mean:.1f}', ha='center', va='bottom')
        
        # Summary comparison (radar chart style)
        categories = ['H-Index\n(Median)', 'Citations\n(Median)', 'Papers/Year\n(Median)']
        ha_values = [results.ha_metrics.h_index_median, 
                     results.ha_metrics.citations_median / 1000,  # Scale citations
                     results.ha_metrics.papers_per_year_median]
        aha_values = [results.aha_metrics.h_index_median,
                      results.aha_metrics.citations_median / 1000,  # Scale citations
                      results.aha_metrics.papers_per_year_median]
        
        x_pos = np.arange(len(categories))
        ax4.plot(x_pos, ha_values, marker='o', linewidth=3, markersize=8, label='HA Authors', color='red')
        ax4.plot(x_pos, aha_values, marker='s', linewidth=3, markersize=8, label='AHA Authors', color='blue')
        ax4.set_title('Productivity Profile Comparison', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylabel('Metric Value\n(Citations in thousands)')
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / "productivity_metrics.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(filepath)
    
    def plot_authorship_patterns(self, results: AnalysisResults) -> Optional[str]:
        """
        Create authorship patterns visualization.
        
        Shows the distribution of first, last, and other author positions
        for HA and AHA groups.
        
        Args:
            results: Analysis results containing authorship data
            
        Returns:
            File path of saved visualization
        """
        if not results.ha_metrics or not results.aha_metrics:
            return None
        
        if not results.ha_metrics.authorship_patterns or not results.aha_metrics.authorship_patterns:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Data for HA authors
        ha_patterns = results.ha_metrics.authorship_patterns
        ha_data = [ha_patterns.first_author_median, ha_patterns.last_author_median, ha_patterns.other_author_median]
        
        # Data for AHA authors
        aha_patterns = results.aha_metrics.authorship_patterns
        aha_data = [aha_patterns.first_author_median, aha_patterns.last_author_median, aha_patterns.other_author_median]
        
        positions = ['First Author', 'Last Author', 'Other Position']
        
        # HA authors pie chart
        colors1 = ['#ff9999', '#66b3ff', '#99ff99']
        wedges1, texts1, autotexts1 = ax1.pie(ha_data, labels=positions, autopct='%1.1f%%',
                                             colors=colors1, startangle=90)
        ax1.set_title(f'HA Authors Authorship Patterns\n(n={results.ha_count})', fontweight='bold')
        
        # AHA authors pie chart
        colors2 = ['#ffcc99', '#ff99cc', '#c2c2f0']
        wedges2, texts2, autotexts2 = ax2.pie(aha_data, labels=positions, autopct='%1.1f%%',
                                             colors=colors2, startangle=90)
        ax2.set_title(f'AHA Authors Authorship Patterns\n(n={results.aha_count})', fontweight='bold')
        
        # Enhance text
        for autotexts in [autotexts1, autotexts2]:
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / "authorship_patterns.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(filepath)
    
    def plot_author_classification(self, results: AnalysisResults) -> Optional[str]:
        """
        Create author classification overview visualization.
        
        Shows the distribution of all author types with percentages.
        
        Args:
            results: Analysis results
            
        Returns:
            File path of saved visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall classification
        categories = ['HA Authors', 'AHA Authors', 'Regular Authors']
        counts = [results.ha_count, results.aha_count, 
                 results.total_unique_authors - results.ep_count]
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        
        # Bar chart
        bars = ax1.bar(categories, counts, color=colors, alpha=0.8)
        ax1.set_title('Author Classification Overview', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Number of Authors')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels and percentages
        total_authors = results.total_unique_authors
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total_authors) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height + total_authors * 0.01,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # EP vs Regular comparison (pie chart)
        ep_data = [results.ep_count, results.total_unique_authors - results.ep_count]
        ep_labels = [f'EP Authors\n({results.ep_percentage:.1f}%)', 
                    f'Regular Authors\n({100-results.ep_percentage:.1f}%)']
        ep_colors = ['#ff6b6b', '#4ecdc4']
        
        wedges, texts, autotexts = ax2.pie(ep_data, labels=ep_labels, autopct='%1.0f',
                                          colors=ep_colors, startangle=90, textprops={'fontsize': 11})
        ax2.set_title('Extremely Productive vs Regular Authors', fontweight='bold', fontsize=14)
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / "author_classification.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(filepath)
    
    def plot_h_index_distribution(self, results: AnalysisResults) -> Optional[str]:
        """
        Create H-index distribution visualization for EP authors.
        
        Args:
            results: Analysis results
            
        Returns:
            File path of saved visualization
        """
        if not results.ep_authors:
            return None
        
        # Extract H-index values
        h_indices = [author.h_index for author in results.ep_authors 
                    if author.h_index is not None and author.h_index > 0]
        
        if not h_indices:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(h_indices, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('H-Index Distribution (EP Authors)', fontweight='bold')
        ax1.set_xlabel('H-Index')
        ax1.set_ylabel('Number of Authors')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_h = np.mean(h_indices)
        median_h = np.median(h_indices)
        ax1.axvline(mean_h, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_h:.1f}')
        ax1.axvline(median_h, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_h:.1f}')
        ax1.legend()
        
        # Box plot comparison
        ha_h_indices = [author.h_index for author in results.ha_authors 
                       if author.h_index is not None and author.h_index > 0]
        aha_h_indices = [author.h_index for author in results.aha_authors 
                        if author.h_index is not None and author.h_index > 0]
        
        box_data = []
        box_labels = []
        
        if ha_h_indices:
            box_data.append(ha_h_indices)
            box_labels.append('HA')
        
        if aha_h_indices:
            box_data.append(aha_h_indices)
            box_labels.append('AHA')
        
        if box_data:
            ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            ax2.set_title('H-Index Comparison by Author Type', fontweight='bold')
            ax2.set_ylabel('H-Index')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / "h_index_distribution.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(filepath)
    
    def plot_collaboration_patterns(self, results: AnalysisResults) -> Optional[str]:
        """
        Create collaboration patterns visualization.
        
        Args:
            results: Analysis results
            
        Returns:
            File path of saved visualization
        """
        if not results.ep_authors:
            return None
        
        # Analyze collaboration patterns
        solo_papers = 0
        small_teams = 0  # 2-5 authors
        medium_teams = 0  # 6-10 authors
        large_teams = 0  # 11+ authors
        total_papers = 0
        
        for author in results.ep_authors:
            for pub in author.publications:
                if pub.total_authors:
                    total_papers += 1
                    if pub.total_authors == 1:
                        solo_papers += 1
                    elif pub.total_authors <= 5:
                        small_teams += 1
                    elif pub.total_authors <= 10:
                        medium_teams += 1
                    else:
                        large_teams += 1
        
        if total_papers == 0:
            return None
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collaboration distribution
        collab_data = [solo_papers, small_teams, medium_teams, large_teams]
        collab_labels = ['Solo\n(1 author)', 'Small Team\n(2-5 authors)', 
                        'Medium Team\n(6-10 authors)', 'Large Team\n(11+ authors)']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        bars = ax1.bar(collab_labels, collab_data, color=colors, alpha=0.8)
        ax1.set_title('Collaboration Patterns in EP Author Publications', fontweight='bold')
        ax1.set_ylabel('Number of Publications')
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, count in zip(bars, collab_data):
            height = bar.get_height()
            percentage = (count / total_papers) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(collab_data) * 0.01,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(collab_data, labels=collab_labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax2.set_title('Collaboration Distribution\n(Percentage of Publications)', fontweight='bold')
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / "collaboration_patterns.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(filepath)
    
    def plot_annual_publication_trends(self, results: AnalysisResults) -> Optional[str]:
        """
        Create annual publication trends visualization.
        
        Args:
            results: Analysis results
            
        Returns:
            File path of saved visualization
        """
        if not results.ep_authors:
            return None
        
        # Count publications by year
        yearly_publications = {year: 0 for year in YEARS}
        
        for author in results.ep_authors:
            for pub in author.publications:
                if pub.year and pub.year in YEARS:
                    yearly_publications[pub.year] += 1
        
        if not any(yearly_publications.values()):
            return None
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        years = list(yearly_publications.keys())
        pub_counts = list(yearly_publications.values())
        
        # Line plot with markers
        ax.plot(years, pub_counts, marker='o', linewidth=3, markersize=8, color='#2E86AB', alpha=0.8)
        ax.fill_between(years, pub_counts, alpha=0.3, color='#2E86AB')
        
        ax.set_title('Annual Publication Trends (EP Authors)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for year, count in zip(years, pub_counts):
            ax.text(year, count + max(pub_counts) * 0.02, str(count), 
                   ha='center', va='bottom', fontweight='bold')
        
        # Add trend line
        z = np.polyfit(years, pub_counts, 1)
        p = np.poly1d(z)
        ax.plot(years, p(years), "--", alpha=0.7, color='red', linewidth=2, label='Trend')
        
        # Calculate and display trend
        slope = z[0]
        trend_text = f"Trend: {'+' if slope > 0 else ''}{slope:.1f} publications/year"
        ax.text(0.02, 0.98, trend_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        ax.legend()
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / "annual_publication_trends.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(filepath)
    
    def create_summary_dashboard(self, results: AnalysisResults) -> Optional[str]:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            results: Analysis results
            
        Returns:
            File path of saved dashboard
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Hyperprolific Author Analysis - Summary Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Key Statistics (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        
        stats_text = f"""
KEY FINDINGS

Total Unique Authors: {results.total_unique_authors:,}
Extremely Productive Authors: {results.ep_count} ({results.ep_percentage:.1f}%)
Hyperprolific (HA): {results.ha_count}
Almost Hyperprolific (AHA): {results.aha_count}

Study Period: {YEARS[0]}-{YEARS[-1]}
Peak Year: {results.temporal_trends.peak_year if results.temporal_trends else 'N/A'}
        """
        
        ax1.text(0.1, 0.9, stats_text, transform=ax1.transAxes, fontsize=14,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
        
        # 2. Geographic Distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if results.geographic_distribution:
            geo_data = results.geographic_distribution.to_dict()
            continents = [cont for cont, count in geo_data.items() if count > 0]
            counts = [count for count in geo_data.values() if count > 0]
            
            if continents and counts:
                colors = sns.color_palette("husl", len(continents))
                wedges, texts, autotexts = ax2.pie(counts, labels=continents, autopct='%1.1f%%',
                                                  colors=colors, startangle=90)
                ax2.set_title('Geographic Distribution', fontweight='bold')
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
        
        # 3. Author Classification (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        categories = ['HA', 'AHA', 'Regular']
        counts = [results.ha_count, results.aha_count, 
                 results.total_unique_authors - results.ep_count]
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        
        bars = ax3.bar(categories, counts, color=colors, alpha=0.8)
        ax3.set_title('Author Classification', fontweight='bold')
        ax3.set_ylabel('Number of Authors')
        
        # Add labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 4. Temporal Trends (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if results.temporal_trends and results.temporal_trends.annual_ep_counts:
            years = sorted(results.temporal_trends.annual_ep_counts.keys())
            ep_counts = [results.temporal_trends.annual_ep_counts.get(year, 0) for year in years]
            
            ax4.plot(years, ep_counts, marker='o', linewidth=3, markersize=6, color='#2ca02c')
            ax4.set_title('EP Authors Over Time', fontweight='bold')
            ax4.set_ylabel('Number of EP Authors')
            ax4.grid(True, alpha=0.3)
        
        # 5. Productivity Comparison (bottom left)
        ax5 = fig.add_subplot(gs[2, :2])
        if results.ha_metrics and results.aha_metrics:
            metrics = ['H-Index', 'Citations (k)', 'Papers/Year']
            ha_values = [results.ha_metrics.h_index_median,
                        results.ha_metrics.citations_median / 1000,
                        results.ha_metrics.papers_per_year_median]
            aha_values = [results.aha_metrics.h_index_median,
                         results.aha_metrics.citations_median / 1000,
                         results.aha_metrics.papers_per_year_median]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax5.bar(x - width/2, ha_values, width, label='HA', color='#d62728', alpha=0.8)
            ax5.bar(x + width/2, aha_values, width, label='AHA', color='#ff7f0e', alpha=0.8)
            ax5.set_title('Productivity Metrics Comparison', fontweight='bold')
            ax5.set_xticks(x)
            ax5.set_xticklabels(metrics)
            ax5.legend()
        
        # 6. Top Productive Authors (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        if results.top_productive_authors:
            top_text = "TOP PRODUCTIVE AUTHORS\n\n"
            for i, (name, papers) in enumerate(results.top_productive_authors[:10], 1):
                top_text += f"{i}. {name}: {papers:.1f} papers/year\n"
            
            ax6.text(0.05, 0.95, top_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.7))
        
        # 7. Study Information (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        study_info = f"""
STUDY METHODOLOGY: Analysis of hyperprolific authors in orthopaedic research ({YEARS[0]}-{YEARS[-1]})
DATA SOURCE: Top 20 orthopaedic journals by CiteScore | THRESHOLDS: HA ≥72 papers/year, AHA 61-72 papers/year
CLASSIFICATION: {results.ep_count} Extremely Productive (EP) authors identified from {results.total_unique_authors:,} total unique authors
        """
        
        ax7.text(0.5, 0.5, study_info, transform=ax7.transAxes, fontsize=12,
                ha='center', va='center', style='italic',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
        
        # Save
        filepath = self.output_dir / "summary_dashboard.png"
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(filepath)
    
    def create_publication_table(self, results: AnalysisResults, filename: str = "summary_table.csv") -> str:
        """
        Create a CSV table with summary statistics.
        
        Args:
            results: Analysis results
            filename: Output filename
            
        Returns:
            File path of saved table
        """
        data = {
            'Metric': [
                'Total Unique Authors',
                'Hyperprolific (HA) Authors',
                'Almost Hyperprolific (AHA) Authors',
                'Extremely Productive (EP) Authors',
                'EP Percentage',
                'Europe EP Authors',
                'Asia EP Authors',
                'Americas EP Authors',
                'HA Median H-Index',
                'AHA Median H-Index',
                'HA Median Citations',
                'AHA Median Citations',
                'Peak Year',
                'Peak Year EP Count'
            ],
            'Value': [
                results.total_unique_authors,
                results.ha_count,
                results.aha_count,
                results.ep_count,
                f"{results.ep_percentage:.1f}%",
                results.geographic_distribution.europe if results.geographic_distribution else 0,
                results.geographic_distribution.asia if results.geographic_distribution else 0,
                results.geographic_distribution.americas if results.geographic_distribution else 0,
                f"{results.ha_metrics.h_index_median:.1f}" if results.ha_metrics else "N/A",
                f"{results.aha_metrics.h_index_median:.1f}" if results.aha_metrics else "N/A",
                f"{results.ha_metrics.citations_median:.0f}" if results.ha_metrics else "N/A",
                f"{results.aha_metrics.citations_median:.0f}" if results.aha_metrics else "N/A",
                results.temporal_trends.peak_year if results.temporal_trends else "N/A",
                results.temporal_trends.peak_year_count if results.temporal_trends else "N/A"
            ]
        }
        
        df = pd.DataFrame(data)
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        
        return str(filepath)

def create_quick_visualization(results: AnalysisResults, output_dir: str = "quick_viz") -> List[str]:
    """
    Quick function to create essential visualizations.
    
    Args:
        results: Analysis results
        output_dir: Output directory
        
    Returns:
        List of created file paths
    """
    viz = VisualizationGenerator(output_dir)
    
    essential_plots = [
        viz.plot_geographic_distribution(results),
        viz.plot_author_classification(results),
        viz.plot_temporal_trends(results)
    ]
    
    return [path for path in essential_plots if path is not None]

if __name__ == "__main__":
    # Test visualization generation
    print("Testing VisualizationGenerator...")
    
    # This would normally use real results
    # For testing, we'd need sample data
    print("VisualizationGenerator ready for use!") 