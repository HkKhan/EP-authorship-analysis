"""
Visualization module for hyperprolific author analysis
Creates charts and plots to visualize the key findings from the paper's results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os

from data_models import AnalysisResults, Author, Publication

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HyperprolificVisualization:
    """
    Creates visualizations for the hyperprolific author analysis
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_geographic_distribution(self, results: AnalysisResults, save: bool = True) -> None:
        """
        Create a pie chart showing geographic distribution of EP authors
        """
        geo_data = results.geographic_distribution.get_percentages()
        
        # Filter out regions with 0%
        regions = [region.title() for region, pct in geo_data.items() if pct > 0]
        percentages = [pct for pct in geo_data.values() if pct > 0]
        counts = [getattr(results.geographic_distribution, region.lower()) 
                 for region in geo_data.keys() if getattr(results.geographic_distribution, region.lower()) > 0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(percentages, labels=regions, autopct='%1.1f%%', 
                                         startangle=90, textprops={'fontsize': 12})
        
        # Add count labels
        for i, (wedge, count) in enumerate(zip(wedges, counts)):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 0.7 * np.cos(np.radians(angle))
            y = 0.7 * np.sin(np.radians(angle))
            ax.text(x, y, f'n={count}', ha='center', va='center', fontweight='bold')
        
        ax.set_title('Geographic Distribution of Extremely Productive (EP) Authors\n2020-2024', 
                    fontsize=14, fontweight='bold', pad=20)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'geographic_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_annual_ep_trends(self, results: AnalysisResults, save: bool = True) -> None:
        """
        Create a line plot showing annual trends in EP author counts
        """
        years = list(results.annual_ep_counts.keys())
        ep_counts = list(results.annual_ep_counts.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot EP trend line
        ax.plot(years, ep_counts, marker='o', linewidth=3, markersize=8, 
               color='#2E86AB', label='EP Authors')
        
        # Add value labels on points
        for year, count in zip(years, ep_counts):
            ax.annotate(f'{count}', (year, count), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontweight='bold')
        
        # Highlight peak year
        peak_year = max(results.annual_ep_counts, key=results.annual_ep_counts.get)
        peak_count = results.annual_ep_counts[peak_year]
        ax.scatter([peak_year], [peak_count], color='red', s=100, zorder=5)
        ax.annotate(f'Peak: {peak_year}', (peak_year, peak_count), 
                   textcoords="offset points", xytext=(0,20), ha='center', 
                   fontweight='bold', color='red')
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of EP Authors', fontsize=12, fontweight='bold')
        ax.set_title('Annual Trends in Extremely Productive (EP) Authors\nOrthopaedic Research 2020-2024', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(years)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'annual_ep_trends.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_productivity_metrics_comparison(self, results: AnalysisResults, save: bool = True) -> None:
        """
        Create box plots comparing H-index and citation metrics between HA and AHA authors
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # H-index comparison
        h_index_data = {
            'HA': [author.h_index for author in results.ha_authors],
            'AHA': [author.h_index for author in results.aha_authors]
        }
        
        bp1 = ax1.boxplot([h_index_data['HA'], h_index_data['AHA']], 
                         labels=['HA Authors', 'AHA Authors'],
                         patch_artist=True, notch=True)
        
        # Color the boxes
        colors = ['#FF6B6B', '#4ECDC4']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('H-index', fontsize=12, fontweight='bold')
        ax1.set_title('H-index Distribution\nHA vs AHA Authors', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistical annotations
        ha_h_median = results.h_index_metrics['HA'].median
        aha_h_median = results.h_index_metrics['AHA'].median
        ax1.text(0.02, 0.98, f'HA Median: {ha_h_median:.0f}\nAHA Median: {aha_h_median:.0f}', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Citations comparison
        citation_data = {
            'HA': [author.total_citations for author in results.ha_authors],
            'AHA': [author.total_citations for author in results.aha_authors]
        }
        
        bp2 = ax2.boxplot([citation_data['HA'], citation_data['AHA']], 
                         labels=['HA Authors', 'AHA Authors'],
                         patch_artist=True, notch=True)
        
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Total Citations', fontsize=12, fontweight='bold')
        ax2.set_title('Citation Distribution\nHA vs AHA Authors', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistical annotations
        ha_cit_median = results.citation_metrics['HA'].median
        aha_cit_median = results.citation_metrics['AHA'].median
        ax2.text(0.02, 0.98, f'HA Median: {ha_cit_median:.0f}\nAHA Median: {aha_cit_median:.0f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'productivity_metrics_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_authorship_positions(self, results: AnalysisResults, save: bool = True) -> None:
        """
        Create bar plots showing authorship position patterns for HA and AHA authors
        """
        from analysis_engine import HyperprolificAnalysisEngine
        
        engine = HyperprolificAnalysisEngine()
        ha_positions = engine.analyze_authorship_positions(results.ha_authors)
        aha_positions = engine.analyze_authorship_positions(results.aha_authors)
        
        positions = ['First Author', 'Last Author', 'Other Positions']
        ha_values = [ha_positions['first_median'], ha_positions['last_median'], ha_positions['middle_median']]
        aha_values = [aha_positions['first_median'], aha_positions['last_median'], aha_positions['middle_median']]
        
        x = np.arange(len(positions))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, ha_values, width, label='HA Authors', 
                      color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, aha_values, width, label='AHA Authors', 
                      color='#4ECDC4', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Authorship Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (Median)', fontsize=12, fontweight='bold')
        ax.set_title('Authorship Position Patterns\nHA vs AHA Authors (Median Percentages)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add IQR information in text box
        info_text = f"""IQR Ranges:
HA First: {ha_positions['first_iqr'][0]:.1f}%-{ha_positions['first_iqr'][1]:.1f}%
HA Last: {ha_positions['last_iqr'][0]:.1f}%-{ha_positions['last_iqr'][1]:.1f}%
AHA First: {aha_positions['first_iqr'][0]:.1f}%-{aha_positions['first_iqr'][1]:.1f}%
AHA Last: {aha_positions['last_iqr'][0]:.1f}%-{aha_positions['last_iqr'][1]:.1f}%"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'authorship_positions.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_top_productive_authors(self, results: AnalysisResults, top_n: int = 10, save: bool = True) -> None:
        """
        Create horizontal bar chart of most productive authors
        """
        from analysis_engine import HyperprolificAnalysisEngine
        
        engine = HyperprolificAnalysisEngine()
        top_authors = engine.identify_most_productive_authors(results.ep_authors, top_n)
        
        names = [author['name'] for author in top_authors]
        publications = [author['total_publications'] for author in top_authors]
        classifications = [author['classification'] for author in top_authors]
        
        # Color by classification
        colors = ['#FF6B6B' if cls == 'HA' else '#4ECDC4' for cls in classifications]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(names, publications, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, pub_count) in enumerate(zip(bars, publications)):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                   f'{pub_count}', va='center', fontweight='bold')
        
        ax.set_xlabel('Total Publications (2020-2024)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Productive Authors in Orthopaedic Research', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FF6B6B', alpha=0.8, label='HA Authors'),
                          Patch(facecolor='#4ECDC4', alpha=0.8, label='AHA Authors')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'top_productive_authors.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ep_duration_distribution(self, results: AnalysisResults, save: bool = True) -> None:
        """
        Create histogram showing distribution of EP status duration
        """
        from analysis_engine import HyperprolificAnalysisEngine
        
        engine = HyperprolificAnalysisEngine()
        consistency_data = engine.analyze_ep_consistency(results.ep_authors)
        
        durations = list(consistency_data['duration_distribution'].keys())
        counts = list(consistency_data['duration_distribution'].values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(durations, counts, color='#45B7D1', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / len(results.ep_authors)) * 100
            ax.annotate(f'{count}\n({percentage:.1f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Duration of EP Status (Years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Authors', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of EP Status Duration\n(2020-2024)', fontsize=14, fontweight='bold')
        ax.set_xticks(durations)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add median line
        median_duration = consistency_data['median_duration']
        ax.axvline(median_duration, color='red', linestyle='--', linewidth=2, 
                  label=f'Median: {median_duration:.1f} years')
        ax.legend()
        
        # Add summary statistics
        one_year_pct = consistency_data['percentage_one_year_only']
        consistent_count = len(consistency_data['consistent_5_year_authors'])
        
        stats_text = f"""Summary Statistics:
• Total EP Authors: {len(results.ep_authors)}
• One year only: {one_year_pct:.1f}%
• All 5 years: {consistent_count} authors
• Median duration: {median_duration:.1f} years"""
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'ep_duration_distribution.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self, results: AnalysisResults, save: bool = True) -> None:
        """
        Create a comprehensive dashboard with multiple visualizations
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Geographic distribution (top left)
        ax1 = plt.subplot(2, 3, 1)
        geo_data = results.geographic_distribution.get_percentages()
        regions = [region.title() for region, pct in geo_data.items() if pct > 0]
        percentages = [pct for pct in geo_data.values() if pct > 0]
        ax1.pie(percentages, labels=regions, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Geographic Distribution', fontweight='bold')
        
        # Annual trends (top middle)
        ax2 = plt.subplot(2, 3, 2)
        years = list(results.annual_ep_counts.keys())
        ep_counts = list(results.annual_ep_counts.values())
        ax2.plot(years, ep_counts, marker='o', linewidth=2, markersize=6)
        ax2.set_title('Annual EP Author Trends', fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('EP Authors')
        ax2.grid(True, alpha=0.3)
        
        # H-index comparison (top right)
        ax3 = plt.subplot(2, 3, 3)
        h_data = [[author.h_index for author in results.ha_authors],
                  [author.h_index for author in results.aha_authors]]
        ax3.boxplot(h_data, labels=['HA', 'AHA'])
        ax3.set_title('H-index Distribution', fontweight='bold')
        ax3.set_ylabel('H-index')
        
        # Authorship positions (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        from analysis_engine import HyperprolificAnalysisEngine
        engine = HyperprolificAnalysisEngine()
        ha_pos = engine.analyze_authorship_positions(results.ha_authors)
        aha_pos = engine.analyze_authorship_positions(results.aha_authors)
        
        positions = ['First', 'Last', 'Other']
        ha_vals = [ha_pos['first_median'], ha_pos['last_median'], ha_pos['middle_median']]
        aha_vals = [aha_pos['first_median'], aha_pos['last_median'], aha_pos['middle_median']]
        
        x = np.arange(len(positions))
        width = 0.35
        ax4.bar(x - width/2, ha_vals, width, label='HA', alpha=0.8)
        ax4.bar(x + width/2, aha_vals, width, label='AHA', alpha=0.8)
        ax4.set_title('Authorship Positions', fontweight='bold')
        ax4.set_ylabel('Percentage (Median)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(positions)
        ax4.legend()
        
        # Top authors (bottom middle)
        ax5 = plt.subplot(2, 3, 5)
        top_authors = engine.identify_most_productive_authors(results.ep_authors, 5)
        names = [author['name'].split()[-1] for author in top_authors]  # Last name only
        pubs = [author['total_publications'] for author in top_authors]
        ax5.barh(names, pubs, alpha=0.8)
        ax5.set_title('Top 5 Most Productive', fontweight='bold')
        ax5.set_xlabel('Publications')
        
        # EP duration (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        consistency_data = engine.analyze_ep_consistency(results.ep_authors)
        durations = list(consistency_data['duration_distribution'].keys())
        counts = list(consistency_data['duration_distribution'].values())
        ax6.bar(durations, counts, alpha=0.8)
        ax6.set_title('EP Duration Distribution', fontweight='bold')
        ax6.set_xlabel('Years')
        ax6.set_ylabel('Authors')
        
        plt.suptitle('Hyperprolific Author Analysis Dashboard\nOrthopaedic Research 2020-2024', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'comprehensive_dashboard.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()

def create_all_visualizations(results: AnalysisResults, output_dir: str = "visualizations") -> None:
    """
    Generate all visualizations for the analysis results
    """
    viz = HyperprolificVisualization(output_dir)
    
    print("Creating visualizations...")
    viz.plot_geographic_distribution(results)
    viz.plot_annual_ep_trends(results)
    viz.plot_productivity_metrics_comparison(results)
    viz.plot_authorship_positions(results)
    viz.plot_top_productive_authors(results)
    viz.plot_ep_duration_distribution(results)
    viz.create_comprehensive_dashboard(results)
    
    print(f"All visualizations saved to {output_dir}/")

if __name__ == "__main__":
    # Test with sample data
    from scopus_data_extractor import create_sample_data
    from analysis_engine import HyperprolificAnalysisEngine
    
    print("Creating sample data and running analysis...")
    publications, authors = create_sample_data()
    
    engine = HyperprolificAnalysisEngine()
    results = engine.run_complete_analysis(publications, authors)
    
    # Create visualizations
    create_all_visualizations(results) 