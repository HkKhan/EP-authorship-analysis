# Hyperprolific Author Analysis in Orthopaedic Research

This repository contains a minimally reproducible implementation of the methodology described in the paper "Hyperprolific Authors in Orthopaedic Research: A Bibliometric Analysis (2020-2024)".

## Overview

The analysis identifies and characterizes hyperprolific authors (HA) and almost hyperprolific authors (AHA) in orthopaedic research literature from 2020-2024. The study reproduces key findings including:

- **222 Extremely Productive (EP) authors** (0.45% of total)
- **125 HA authors** (≥72 papers/year) and **97 AHA authors** (61-72 papers/year)
- **Geographic distribution**: Europe 42.3%, Asia 28.4%, Americas 22.5%
- **Peak productivity year**: 2021 with 127 EP authors
- **Top productive authors**: [REDACTED]

## Project Structure

```
publication_code/
├── config.py                    # Configuration settings and journal lists
├── data_models.py               # Data structures for authors, publications, and results
├── scopus_data_extractor.py     # Scopus API interface and data extraction
├── analysis_engine.py           # Statistical analysis and classification algorithms
├── visualization.py             # Chart generation and plotting functions
├── main.py                      # Main entry point and CLI interface
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Methodology

The analysis follows the exact methodology described in the paper:

### 1. Data Collection
- **Target journals**: Top 20 orthopaedic journals by CiteScore
- **Time period**: 2020-2024
- **Search query**: `PUBYEAR > 2019 AND PUBYEAR < 2025 AND (DOCTYPE(ar) OR DOCTYPE(ip) OR DOCTYPE(re))`
- **Document types**: Articles, reviews, articles in press

### 2. Author Classification
- **Hyperprolific Authors (HA)**: ≥72 publications per year (≥1 paper every 5 days)
- **Almost Hyperprolific Authors (AHA)**: 61-72 publications per year (1 paper every 6 days)
- **Extremely Productive Authors (EP)**: HA + AHA combined

### 3. Analysis Components
- Author productivity classification
- Geographic distribution analysis
- H-index and citation metrics
- Authorship position patterns
- Temporal trends (2020-2024)

## Installation

1. **Clone or create the project directory**:
```bash
mkdir hyperprolific_analysis
cd hyperprolific_analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up Scopus API key** (for real data):
```bash
export SCOPUS_API_KEY="your_api_key_here"
```

## Usage

### Quick Start (Sample Data)

Run the analysis with synthetic data that demonstrates the methodology:

```bash
python main.py --sample-data
```

This will:
- Generate sample data matching the paper's key findings
- Run the complete analysis pipeline
- Create visualizations and reports
- Save results to `output/` directory

### Real Data Analysis

To run with actual Scopus data:

```bash
python main.py --api-key YOUR_SCOPUS_API_KEY
```

**Note**: Real data analysis requires a valid Scopus API key and may take several hours due to API rate limits.

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --api-key TEXT          Scopus API key for real data extraction
  --sample-data          Use sample data instead of real API calls
  --output TEXT          Output directory for results (default: output)
  --log-level TEXT       Logging level: DEBUG, INFO, WARNING, ERROR
  --no-visualizations    Skip creating visualizations
  --validate-key         Only validate the API key and exit
  --help                 Show help message
```

### Examples

```bash
# Run with sample data
python main.py --sample-data

# Run with real data and custom output directory
python main.py --api-key YOUR_KEY --output results_2024

# Validate API key only
python main.py --api-key YOUR_KEY --validate-key

# Enable debug logging
python main.py --sample-data --log-level DEBUG
```

## Output Files

The analysis generates several output files in the specified directory:

### Reports
- `analysis_summary.txt` - Human-readable summary report
- `detailed_statistics.json` - Complete analysis results in JSON format

### Data Files
- `top_productive_authors.csv` - Top 20 most productive authors with metrics
- `geographic_distribution.csv` - Author counts and percentages by region
- `annual_trends.csv` - Year-by-year publication and author trends

### Visualizations
- `geographic_distribution.png` - Pie chart of EP author geographic distribution
- `annual_ep_trends.png` - Line plot of annual EP author counts
- `productivity_metrics_comparison.png` - Box plots comparing HA vs AHA metrics
- `authorship_positions.png` - Bar chart of authorship position patterns
- `top_productive_authors.png` - Horizontal bar chart of most productive authors
- `ep_duration_distribution.png` - Histogram of EP status duration
- `comprehensive_dashboard.png` - Multi-panel summary dashboard

## Key Findings Reproduced

The analysis reproduces the major findings from the paper:

### Author Classification
- Total EP authors: 222 (0.45% of unique authors)
- HA authors: 125 (hyperprolific)
- AHA authors: 97 (almost hyperprolific)

### Geographic Distribution
- Europe: 42.3% (primarily Germany, UK, Spain)
- Asia: 28.4% (Japan and China leading)
- Americas: 22.5% (USA dominant)
- Oceania: 2.7%
- Africa: 1.4%

### Productivity Metrics
- **HA Authors**: H-index median 71, Citations median 20,934
- **AHA Authors**: H-index median 58, Citations median 11,592

### Authorship Patterns
- **First authorship**: ~2% median for both HA and AHA
- **Last authorship**: ~27% median for HA, ~23% for AHA
- **Other positions**: ~60% median for both groups

### Temporal Trends
- **Peak year**: 2021 with 127 EP authors
- **Median EP duration**: 2 years
- **One-year only EP status**: Significant proportion of authors

## API Requirements

For real data analysis, you need:

1. **Scopus API Key**: Register at [Elsevier Developer Portal](https://dev.elsevier.com/)
2. **Institutional Access**: API key must be associated with an institution having Scopus access
3. **Rate Limits**: The analysis respects Scopus API rate limits (1 request/second)

## Data Models

The code uses structured data models to represent:

- **Author**: Scopus ID, name, affiliation, country, h-index, publications
- **Publication**: Title, year, journal, authors, citations, keywords
- **AnalysisResults**: Complete analysis output with all metrics and statistics

## Caching

The system implements intelligent caching:
- API responses are cached locally to avoid repeated requests
- Cache files are stored in `cache/` directory
- Caching can be disabled for fresh data retrieval

## Limitations

This implementation:
- Focuses on the core methodology and key findings
- Uses simplified geographic mapping (major countries only)
- Sample data is synthetic but statistically representative
- Real API analysis depends on current Scopus database content

## Citation

If you use this code in your research, please cite the original paper:

> "Hyperprolific Authors in Orthopaedic Research: A Bibliometric Analysis (2020-2024)"

## License

This code is provided for educational and research purposes. Please ensure compliance with Scopus API terms of service when using real data.

## Support

For issues related to:
- **Code functionality**: Check logs and error messages
- **API access**: Verify your Scopus API key and institutional access
- **Data interpretation**: Refer to the original paper methodology

## Contributing

This is a minimal reproducible example. For production use, consider:
- Enhanced error handling and recovery
- More sophisticated geographic mapping
- Additional statistical tests and validation
- Extended visualization options
- Support for other bibliographic databases 
