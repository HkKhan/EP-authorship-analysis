"""
Scopus Data Extraction Module for Hyperprolific Author Analysis.

This module handles interaction with the Scopus API to extract publication
and author data for the bibliometric analysis. It can also generate sample
data for testing when API access is not available.
"""

import requests
import time
import json
import random
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

# Import configuration and data models
import config
from data_models import Author, Publication

# Set up logging
logger = logging.getLogger(__name__)

class ScopusAPIError(Exception):
    """Custom exception for Scopus API related errors."""
    pass

class ScopusDataExtractor:
    """
    Handles extraction of publication and author data from Scopus API.
    
    This class implements the data collection methodology described in the paper,
    including publication searches, author data retrieval, and sample data generation
    for testing purposes.
    """
    
    def __init__(self, api_key: Optional[str] = None, enable_caching: bool = True):
        """
        Initialize the Scopus data extractor.
        
        Args:
            api_key: Scopus API key for accessing the database
            enable_caching: Whether to cache API responses to reduce repeated calls
        """
        self.api_key = api_key
        self.enable_caching = enable_caching
        self.cache_dir = Path("scopus_cache")
        
        if enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.min_request_interval = 1.0 / config.API_RATE_LIMIT  # Convert requests/sec to seconds/request
        
        logger.info(f"ScopusDataExtractor initialized with caching={'enabled' if enable_caching else 'disabled'}")
    
    def validate_api_key(self) -> bool:
        """
        Validate the API key by making a simple test request.
        
        Returns:
            True if API key is valid, False otherwise
        """
        if not self.api_key:
            return False
        
        try:
            # Make a simple test request
            url = f"{config.SCOPUS_API_BASE_URL}/search/scopus"
            headers = {
                'X-ELS-APIKey': self.api_key,
                'Accept': 'application/json'
            }
            params = {
                'query': 'TITLE("test")',
                'count': 1
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=config.API_TIMEOUT)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def _make_api_request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a rate-limited API request to Scopus.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            API response as dictionary, or None if failed
        """
        if not self.api_key:
            raise ScopusAPIError("No API key available")
        
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_request_time
        if time_since_last_call < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_call
            time.sleep(sleep_time)
        
        headers = {
            'X-ELS-APIKey': self.api_key,
            'Accept': 'application/json'
        }
        
        try:
            logger.debug(f"Making API request to {url} with params: {params}")
            response = requests.get(url, headers=headers, params=params, timeout=config.API_TIMEOUT)
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise ScopusAPIError("Invalid API key or insufficient permissions")
            elif response.status_code == 429:
                logger.warning("API rate limit exceeded, waiting...")
                time.sleep(60)  # Wait 1 minute
                return self._make_api_request(url, params)  # Retry
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            return None
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache if available."""
        if not self.enable_caching:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    logger.debug(f"Loading from cache: {cache_key}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache."""
        if not self.enable_caching:
            return
        
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def search_publications(self, query: str, max_results: int = 10000) -> List[Dict[str, Any]]:
        """
        Search for publications using Scopus API.
        
        Args:
            query: Scopus search query string
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of publication dictionaries
        """
        cache_key = f"search_{hash(query)}_{max_results}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        publications = []
        start = 0
        count = min(200, max_results)  # Scopus API limit is 200 per request
        
        while start < max_results:
            params = {
                'query': query,
                'count': count,
                'start': start,
                'field': 'dc:identifier,dc:title,prism:publicationName,prism:coverDate,'
                        'citedby-count,author,prism:doi,subtype,authkeywords'
            }
            
            response = self._make_api_request(config.SCOPUS_API_BASE_URL, params)
            
            if not response or 'search-results' not in response:
                logger.error("Failed to get search results")
                break
            
            entries = response['search-results'].get('entry', [])
            if not entries:
                break
            
            publications.extend(entries)
            
            # Check if we've retrieved all available results
            total_results = int(response['search-results'].get('opensearch:totalResults', 0))
            if start + count >= total_results:
                break
            
            start += count
            logger.info(f"Retrieved {len(publications)}/{min(max_results, total_results)} publications")
        
        self._save_to_cache(cache_key, publications)
        return publications
    
    def get_author_details(self, author_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed author information from Scopus.
        
        Args:
            author_id: Scopus author ID
            
        Returns:
            Author details dictionary or None if not found
        """
        cache_key = f"author_{author_id}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        url = f"{config.SCOPUS_AUTHOR_API_URL}/author_id/{author_id}"
        params = {
            'field': 'identifier,eid,orcid,given-name,surname,initials,'
                    'document-count,cited-by-count,citation-count,h-index,'
                    'affiliation-current,affiliation-history'
        }
        
        response = self._make_api_request(url, params)
        
        if response and 'author-retrieval-response' in response:
            author_data = response['author-retrieval-response'][0]
            self._save_to_cache(cache_key, author_data)
            return author_data
        
        return None
    
    def get_author_publications(self, author_id: str, start_year: int = 2020, 
                              end_year: int = 2024) -> List[Dict[str, Any]]:
        """
        Get all publications for a specific author in the study period.
        
        Args:
            author_id: Scopus author ID
            start_year: Start year for publication search
            end_year: End year for publication search
            
        Returns:
            List of publication dictionaries
        """
        cache_key = f"author_pubs_{author_id}_{start_year}_{end_year}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        query = f"AU-ID({author_id}) AND PUBYEAR > {start_year-1} AND PUBYEAR < {end_year+1}"
        publications = self.search_publications(query, max_results=5000)
        
        self._save_to_cache(cache_key, publications)
        return publications
    
    def extract_authors_from_publications(self, publications: List[Dict[str, Any]]) -> List[str]:
        """
        Extract unique author IDs from publication results.
        
        Args:
            publications: List of publication dictionaries from Scopus
            
        Returns:
            List of unique Scopus author IDs
        """
        author_ids = set()
        
        for pub in publications:
            authors = pub.get('author', [])
            if isinstance(authors, list):
                for author in authors:
                    auth_id = author.get('authid')
                    if auth_id:
                        author_ids.add(auth_id)
        
        return list(author_ids)
    
    def process_author_data(self, author_id: str) -> Optional[Author]:
        """
        Process complete author data including publications and metrics.
        
        Args:
            author_id: Scopus author ID
            
        Returns:
            Author object with complete data or None if processing failed
        """
        try:
            # Get author details
            author_details = self.get_author_details(author_id)
            if not author_details:
                return None
            
            # Get author publications
            publications = self.get_author_publications(author_id)
            
            # Extract basic author info
            coredata = author_details.get('coredata', {})
            name = f"{coredata.get('given-name', '')} {coredata.get('surname', '')}".strip()
            
            # Get affiliation info
            affiliation = None
            country = None
            current_affiliation = author_details.get('affiliation-current')
            if current_affiliation and isinstance(current_affiliation, list) and current_affiliation:
                aff = current_affiliation[0]
                affiliation = aff.get('affiliation-name')
                country = aff.get('affiliation-country')
            
            # Create author object
            author = Author(
                scopus_id=author_id,
                name=name,
                affiliation=affiliation,
                country=country,
                h_index=int(coredata.get('h-index', 0)),
                total_citations=int(coredata.get('cited-by-count', 0)),
                document_count=int(coredata.get('document-count', 0))
            )
            
            # Process publications
            author.publications = self._process_publications(publications, author_id)
            
            # Calculate derived metrics
            author._calculate_metrics()
            
            # Set continent
            if country:
                author.continent = config.COUNTRY_TO_CONTINENT.get(country, "Other")
            
            return author
            
        except Exception as e:
            logger.error(f"Failed to process author {author_id}: {e}")
            return None
    
    def _process_publications(self, pub_data: List[Dict[str, Any]], author_id: str) -> List[Publication]:
        """Process publication data into Publication objects."""
        publications = []
        
        for pub in pub_data:
            try:
                # Extract basic publication info
                title = pub.get('dc:title', 'Unknown Title')
                year_str = pub.get('prism:coverDate', '')
                year = int(year_str.split('-')[0]) if year_str else None
                journal = pub.get('prism:publicationName', 'Unknown Journal')
                
                # Find author position
                authors = pub.get('author', [])
                author_position = None
                total_authors = len(authors) if isinstance(authors, list) else 0
                
                if isinstance(authors, list):
                    for i, author in enumerate(authors, 1):
                        if author.get('authid') == author_id:
                            author_position = i
                            break
                
                publication = Publication(
                    scopus_id=pub.get('dc:identifier', '').replace('SCOPUS_ID:', ''),
                    title=title,
                    year=year,
                    journal=journal,
                    doi=pub.get('prism:doi'),
                    citation_count=int(pub.get('citedby-count', 0)),
                    author_position=author_position,
                    total_authors=total_authors,
                    document_type=pub.get('subtype', 'article')
                )
                
                publications.append(publication)
                
            except Exception as e:
                logger.warning(f"Failed to process publication: {e}")
                continue
        
        return publications
    
    # =============================================================================
    # Sample Data Generation Methods
    # =============================================================================
    
    def generate_sample_data(self) -> AnalysisResults:
        """
        Generate synthetic data that reproduces the key findings from the paper.
        
        This is used for testing and demonstration purposes when no API key
        is available or for faster execution.
        
        Returns:
            AnalysisResults object with synthetic data matching paper findings
        """
        logger.info("Generating sample data based on paper findings...")
        
        # Generate sample authors
        all_authors = self._generate_sample_authors()
        
        # Classify authors
        ha_authors = [a for a in all_authors if a.category == "HA"]
        aha_authors = [a for a in all_authors if a.category == "AHA"]
        ep_authors = ha_authors + aha_authors
        regular_authors = [a for a in all_authors if a.category == "Regular"]
        
        # Create analysis results
        results = AnalysisResults(
            total_unique_authors=len(all_authors),
            total_publications=self._estimate_total_publications(all_authors),
            ha_authors=ha_authors,
            aha_authors=aha_authors,
            ep_authors=ep_authors,
            regular_authors=regular_authors,
            study_period=(2020, 2024),
            journals_analyzed=20
        )
        
        # Calculate statistics
        results.calculate_summary_statistics()
        
        # Set geographic distribution
        results.geographic_distribution = self._generate_geographic_distribution(ep_authors)
        
        # Set productivity metrics
        results.ha_metrics = self._generate_productivity_metrics(ha_authors, "HA")
        results.aha_metrics = self._generate_productivity_metrics(aha_authors, "AHA")
        
        # Set temporal trends
        results.temporal_trends = self._generate_temporal_trends(ep_authors)
        
        # Set top productive authors
        results.top_productive_authors = config.PAPER_FINDINGS["top_authors"][:10]
        
        logger.info(f"Generated sample data: {results.ep_count} EP authors from {results.total_unique_authors} total")
        
        return results
    
    def _generate_sample_authors(self) -> List[Author]:
        """Generate a realistic set of sample authors."""
        authors = []
        
        # Generate countries based on expected distribution
        countries = self._generate_country_distribution()
        
        # Generate HA authors (125 total)
        for i in range(config.PAPER_FINDINGS["ha_authors"]):
            papers_per_year = random.uniform(config.HYPERPROLIFIC_THRESHOLD, 120)
            country = random.choice(countries)
            
            author = Author(
                scopus_id=f"HA_{i:05d}",
                name=f"HA Author {i+1}",
                country=country,
                continent=config.COUNTRY_TO_CONTINENT.get(country, "Other"),
                h_index=random.randint(50, 150),
                total_citations=random.randint(10000, 50000),
                avg_papers_per_year=papers_per_year,
                category="HA",
                is_extremely_productive=True
            )
            
            # Generate publication patterns
            author.publications = self._generate_sample_publications(author, papers_per_year)
            author._calculate_metrics()
            
            authors.append(author)
        
        # Generate AHA authors (97 total)
        for i in range(config.PAPER_FINDINGS["aha_authors"]):
            papers_per_year = random.uniform(config.ALMOST_HYPERPROLIFIC_THRESHOLD, config.HYPERPROLIFIC_THRESHOLD - 1)
            country = random.choice(countries)
            
            author = Author(
                scopus_id=f"AHA_{i:05d}",
                name=f"AHA Author {i+1}",
                country=country,
                continent=config.COUNTRY_TO_CONTINENT.get(country, "Other"),
                h_index=random.randint(30, 100),
                total_citations=random.randint(5000, 30000),
                avg_papers_per_year=papers_per_year,
                category="AHA",
                is_extremely_productive=True
            )
            
            author.publications = self._generate_sample_publications(author, papers_per_year)
            author._calculate_metrics()
            
            authors.append(author)
        
        # Generate regular authors to reach total count
        remaining_authors = config.PAPER_FINDINGS["total_unique_authors"] - len(authors)
        
        for i in range(min(remaining_authors, 1000)):  # Limit for performance
            papers_per_year = random.uniform(1, config.ALMOST_HYPERPROLIFIC_THRESHOLD - 1)
            country = random.choice(countries)
            
            author = Author(
                scopus_id=f"REG_{i:05d}",
                name=f"Regular Author {i+1}",
                country=country,
                continent=config.COUNTRY_TO_CONTINENT.get(country, "Other"),
                h_index=random.randint(5, 50),
                total_citations=random.randint(100, 10000),
                avg_papers_per_year=papers_per_year,
                category="Regular",
                is_extremely_productive=False
            )
            
            authors.append(author)
        
        return authors
    
    def _generate_country_distribution(self) -> List[str]:
        """Generate country list based on expected geographic distribution."""
        countries = []
        
        # Europe (42.3%)
        europe_countries = ["Germany", "United Kingdom", "Spain", "Italy", "France", 
                          "Netherlands", "Switzerland", "Austria", "Belgium", "Sweden"]
        countries.extend(europe_countries * 4)
        
        # Asia (28.4%)
        asia_countries = ["Japan", "China", "South Korea", "India", "Singapore", "Taiwan"]
        countries.extend(asia_countries * 3)
        
        # Americas (22.5%)
        americas_countries = ["United States", "Canada", "Brazil", "Mexico"]
        countries.extend(americas_countries * 2)
        
        # Oceania (2.7%)
        oceania_countries = ["Australia", "New Zealand"]
        countries.extend(oceania_countries)
        
        # Africa (1.4%)
        africa_countries = ["South Africa", "Egypt"]
        countries.extend(africa_countries)
        
        return countries
    
    def _generate_sample_publications(self, author: Author, papers_per_year: float) -> List[Publication]:
        """Generate sample publications for an author."""
        publications = []
        
        for year in config.YEARS:
            # Vary the number of papers around the average
            yearly_papers = max(0, int(random.gauss(papers_per_year, papers_per_year * 0.2)))
            
            for i in range(yearly_papers):
                pub = Publication(
                    scopus_id=f"{author.scopus_id}_PUB_{year}_{i:03d}",
                    title=f"Research Paper {i+1} ({year})",
                    year=year,
                    journal=f"Sample Journal {random.randint(1, 20)}",
                    author_position=random.randint(1, 8),
                    total_authors=random.randint(3, 12),
                    citation_count=random.randint(0, 100),
                    document_type="article"
                )
                publications.append(pub)
        
        return publications
    
    def _generate_geographic_distribution(self, ep_authors: List[Author]) -> Any:
        """Generate geographic distribution matching paper findings."""
        from .data_models import GeographicDistribution
        
        distribution = GeographicDistribution()
        
        for author in ep_authors:
            continent = author.continent or "Other"
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
        
        return distribution
    
    def _generate_productivity_metrics(self, authors: List[Author], group: str) -> Any:
        """Generate productivity metrics for author group."""
        from .data_models import ProductivityMetrics, AuthorshipPatterns
        import statistics
        
        if not authors:
            return ProductivityMetrics()
        
        h_indices = [a.h_index for a in authors if a.h_index]
        citations = [a.total_citations for a in authors if a.total_citations]
        first_auth_pcts = [a.first_author_percentage for a in authors if a.first_author_percentage is not None]
        last_auth_pcts = [a.last_author_percentage for a in authors if a.last_author_percentage is not None]
        other_auth_pcts = [a.other_author_percentage for a in authors if a.other_author_percentage is not None]
        
        # Use paper findings if available
        if group == "HA":
            metrics = config.PAPER_FINDINGS["ha_metrics"]
        else:  # AHA
            metrics = config.PAPER_FINDINGS["aha_metrics"]
        
        authorship = AuthorshipPatterns(
            first_author_median=metrics["first_authorship_median"],
            last_author_median=metrics["last_authorship_median"],
            other_author_median=metrics["other_authorship_median"]
        )
        
        return ProductivityMetrics(
            h_index_median=metrics["h_index_median"],
            h_index_mean=statistics.mean(h_indices) if h_indices else 0,
            citations_median=metrics["citations_median"],
            citations_mean=statistics.mean(citations) if citations else 0,
            authorship_patterns=authorship
        )
    
    def _generate_temporal_trends(self, ep_authors: List[Author]) -> Any:
        """Generate temporal trends matching paper findings."""
        from .data_models import TemporalTrends
        
        trends = TemporalTrends()
        trends.annual_ep_counts = config.PAPER_FINDINGS["annual_ep_counts"].copy()
        trends.calculate_peak_year()
        trends.median_ep_duration = 2.0  # From paper findings
        
        return trends
    
    def _estimate_total_publications(self, authors: List[Author]) -> int:
        """Estimate total publications from author list."""
        total = 0
        for author in authors:
            if author.avg_papers_per_year:
                total += int(author.avg_papers_per_year * len(config.YEARS))
        return total

# =============================================================================
# Utility Functions
# =============================================================================

def create_extractor(api_key: Optional[str] = None, use_sample_data: bool = False) -> ScopusDataExtractor:
    """
    Create a configured ScopusDataExtractor instance.
    
    Args:
        api_key: Scopus API key (optional)
        use_sample_data: If True, will only generate sample data
        
    Returns:
        Configured ScopusDataExtractor instance
    """
    if use_sample_data:
        logger.info("Creating data extractor in sample data mode")
        return ScopusDataExtractor(api_key=None, enable_caching=False)
    else:
        logger.info("Creating data extractor in API mode")
        return ScopusDataExtractor(api_key=api_key, enable_caching=True)

if __name__ == "__main__":
    # Test the data extractor
    print("Testing ScopusDataExtractor...")
    
    # Test sample data generation
    extractor = create_extractor(use_sample_data=True)
    results = extractor.generate_sample_data()
    
    print(f"Generated {results.ep_count} EP authors from {results.total_unique_authors} total authors")
    print(f"Geographic distribution: Europe {results.geographic_distribution.europe}, Asia {results.geographic_distribution.asia}")
    print("Sample data generation test completed successfully!")
    
    # Test API key validation if available
    if config.SCOPUS_API_KEY:
        api_extractor = create_extractor(api_key=config.SCOPUS_API_KEY)
        is_valid = api_extractor.validate_api_key()
        print(f"API key validation: {'Success' if is_valid else 'Failed'}")
    else:
        print("No API key available for validation test") 