"""
Scopus data extraction module
Implements the methodology described in the paper for retrieving publication data
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import os
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import asdict
import logging

from config import *
from data_models import Publication, Author

class ScopusDataExtractor:
    """
    Handles data extraction from Scopus API following the paper's methodology:
    1. Search for articles in top 20 orthopaedic journals (2020-2024)
    2. Extract all author IDs from these articles
    3. Retrieve comprehensive publication histories for each author
    """
    
    def __init__(self, api_key: str, enable_caching: bool = True):
        self.api_key = api_key
        self.enable_caching = enable_caching
        self.session = requests.Session()
        self.session.headers.update({
            'X-ELS-APIKey': api_key,
            'Accept': 'application/json'
        })
        
        # Setup caching
        if enable_caching and not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Generate cache file path"""
        return os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if available"""
        if not self.enable_caching:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache"""
        if not self.enable_caching:
            return
        
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def search_journal_articles(self, journal_name: str, start_year: int, end_year: int) -> List[Dict]:
        """
        Search for articles in a specific journal within the time range
        Implements the Scopus search query from the paper:
        PUBYEAR > 2019 AND PUBYEAR < 2025 AND (DOCTYPE (ar) OR DOCTYPE (ip) OR DOCTYPE (re)) AND SOURCE-ID
        """
        cache_key = f"journal_search_{journal_name.replace(' ', '_')}_{start_year}_{end_year}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Build search query
        query_parts = [
            f"PUBYEAR > {start_year - 1}",
            f"PUBYEAR < {end_year + 1}",
            "(DOCTYPE(ar) OR DOCTYPE(ip) OR DOCTYPE(re))",
            f'SRCTYPE(j) AND SRCTITLE("{journal_name}")'
        ]
        query = " AND ".join(query_parts)
        
        # Search parameters
        params = {
            'query': query,
            'count': 200,  # Maximum results per request
            'start': 0,
            'field': 'dc:identifier,dc:title,prism:publicationName,prism:coverDate,author,citedby-count,prism:doi,prism:aggregationType'
        }
        
        all_articles = []
        total_results = None
        
        while True:
            self._rate_limit()
            
            try:
                response = self.session.get(SCOPUS_BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract search results
                search_results = data.get('search-results', {})
                entries = search_results.get('entry', [])
                
                if total_results is None:
                    total_results = int(search_results.get('opensearch:totalResults', 0))
                    self.logger.info(f"Found {total_results} articles for {journal_name}")
                
                if not entries:
                    break
                
                all_articles.extend(entries)
                
                # Check if we have all results
                if len(all_articles) >= total_results:
                    break
                
                # Update start parameter for next page
                params['start'] += params['count']
                
            except requests.RequestException as e:
                self.logger.error(f"Error searching {journal_name}: {e}")
                break
        
        self._save_to_cache(cache_key, all_articles)
        return all_articles
    
    def extract_author_ids_from_articles(self, articles: List[Dict]) -> Set[str]:
        """Extract unique Scopus author IDs from article list"""
        author_ids = set()
        
        for article in articles:
            authors = article.get('author', [])
            if isinstance(authors, list):
                for author in authors:
                    author_id = author.get('@auid')
                    if author_id:
                        author_ids.add(author_id)
        
        return author_ids
    
    def get_author_publications(self, author_id: str) -> List[Dict]:
        """
        Retrieve comprehensive publication history for an author
        This includes publications outside the orthopaedic domain as mentioned in the paper
        """
        cache_key = f"author_pubs_{author_id}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Search for all publications by this author
        query = f"AU-ID({author_id})"
        params = {
            'query': query,
            'count': 200,
            'start': 0,
            'field': 'dc:identifier,dc:title,prism:publicationName,prism:coverDate,author,citedby-count,prism:doi,authkeywords,prism:aggregationType'
        }
        
        all_publications = []
        
        while True:
            self._rate_limit()
            
            try:
                response = self.session.get(SCOPUS_BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                search_results = data.get('search-results', {})
                entries = search_results.get('entry', [])
                
                if not entries:
                    break
                
                # Filter for included document types
                filtered_entries = []
                for entry in entries:
                    doc_type = entry.get('prism:aggregationType', '').lower()
                    subtype = entry.get('subtypeDescription', '').lower()
                    
                    # Include articles, reviews, and articles in press
                    if any(included_type in doc_type or included_type in subtype 
                           for included_type in INCLUDED_DOCTYPES):
                        filtered_entries.append(entry)
                
                all_publications.extend(filtered_entries)
                
                # Check if we have all results
                total_results = int(search_results.get('opensearch:totalResults', 0))
                if len(all_publications) >= total_results:
                    break
                
                params['start'] += params['count']
                
            except requests.RequestException as e:
                self.logger.error(f"Error getting publications for author {author_id}: {e}")
                break
        
        self._save_to_cache(cache_key, all_publications)
        return all_publications
    
    def get_author_profile(self, author_id: str) -> Optional[Dict]:
        """Get author profile information including h-index and affiliation"""
        cache_key = f"author_profile_{author_id}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        self._rate_limit()
        
        try:
            url = f"https://api.elsevier.com/content/author/author_id/{author_id}"
            params = {'field': 'identifier,indexed-name,given-name,surname,affiliation,h-index,document-count,cited-by-count'}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            author_profile = data.get('author-retrieval-response', [{}])[0] if isinstance(data.get('author-retrieval-response'), list) else data.get('author-retrieval-response', {})
            
            self._save_to_cache(cache_key, author_profile)
            return author_profile
            
        except requests.RequestException as e:
            self.logger.error(f"Error getting profile for author {author_id}: {e}")
            return None
    
    def extract_complete_dataset(self) -> Tuple[List[Publication], List[Author]]:
        """
        Execute the complete data extraction process following the paper's methodology:
        1. Search articles in top 20 orthopaedic journals (2020-2024)
        2. Extract all author IDs
        3. Get comprehensive publication histories
        4. Build Author and Publication objects
        """
        self.logger.info("Starting complete dataset extraction...")
        
        # Step 1: Search articles in top journals
        all_articles = []
        all_author_ids = set()
        
        for journal in TOP_ORTHOPAEDIC_JOURNALS:
            self.logger.info(f"Searching {journal}...")
            articles = self.search_journal_articles(journal, STUDY_START_YEAR, STUDY_END_YEAR)
            all_articles.extend(articles)
            
            # Extract author IDs from these articles
            author_ids = self.extract_author_ids_from_articles(articles)
            all_author_ids.update(author_ids)
            
            self.logger.info(f"Found {len(articles)} articles and {len(author_ids)} unique authors in {journal}")
        
        self.logger.info(f"Total: {len(all_articles)} articles, {len(all_author_ids)} unique authors")
        
        # Step 2: Get comprehensive publication histories for all authors
        authors = []
        publications = []
        
        for i, author_id in enumerate(all_author_ids):
            if i % 100 == 0:
                self.logger.info(f"Processing author {i+1}/{len(all_author_ids)}")
            
            # Get author profile
            profile = self.get_author_profile(author_id)
            if not profile:
                continue
            
            # Get all publications by this author
            author_pubs = self.get_author_publications(author_id)
            
            # Convert to Publication objects
            pub_objects = []
            for pub_data in author_pubs:
                try:
                    # Extract publication year
                    cover_date = pub_data.get('prism:coverDate', '')
                    year = int(cover_date.split('-')[0]) if cover_date else 0
                    
                    # Skip publications outside study period for main analysis
                    if year < STUDY_START_YEAR or year > STUDY_END_YEAR:
                        continue
                    
                    # Extract author list
                    authors_list = []
                    for author in pub_data.get('author', []):
                        auth_id = author.get('@auid')
                        if auth_id:
                            authors_list.append(auth_id)
                    
                    # Create Publication object
                    publication = Publication(
                        scopus_id=pub_data.get('dc:identifier', '').replace('SCOPUS_ID:', ''),
                        title=pub_data.get('dc:title', ''),
                        year=year,
                        journal=pub_data.get('prism:publicationName', ''),
                        authors=authors_list,
                        keywords=pub_data.get('authkeywords', '').split(' | ') if pub_data.get('authkeywords') else [],
                        citation_count=int(pub_data.get('citedby-count', 0)),
                        document_type=pub_data.get('prism:aggregationType', ''),
                        source_id=pub_data.get('source-id', '')
                    )
                    
                    pub_objects.append(publication)
                    
                except (ValueError, KeyError) as e:
                    continue
            
            # Create Author object
            try:
                # Extract name and affiliation from profile
                coredata = profile.get('coredata', {})
                name = coredata.get('indexed-name', 'Unknown')
                
                # Get current affiliation
                affiliation_current = profile.get('affiliation-current', {})
                if isinstance(affiliation_current, list) and affiliation_current:
                    affiliation_current = affiliation_current[0]
                
                affiliation = affiliation_current.get('affiliation-name', 'Unknown')
                country = affiliation_current.get('affiliation-country', 'Unknown')
                
                # Get metrics
                h_index = int(profile.get('h-index', 0))
                total_citations = int(profile.get('citeInfo', {}).get('citedby-count', 0))
                
                author = Author(
                    scopus_id=author_id,
                    name=name,
                    affiliation=affiliation,
                    country=country,
                    h_index=h_index,
                    total_citations=total_citations,
                    publications=pub_objects
                )
                
                authors.append(author)
                publications.extend(pub_objects)
                
            except (ValueError, KeyError) as e:
                self.logger.warning(f"Error processing author {author_id}: {e}")
                continue
        
        self.logger.info(f"Extraction complete: {len(authors)} authors, {len(publications)} publications")
        return publications, authors

# Example usage and testing functions
def create_sample_data() -> Tuple[List[Publication], List[Author]]:
    """
    Create sample data for testing when API access is not available
    This simulates the data structure and key findings from the paper
    """
    # Create sample publications
    sample_publications = []
    sample_authors = []
    
    # Create hyperprolific authors based on paper findings
    top_ha_authors = [
        ("Lip G.Y.H.", "United Kingdom", 1174, 86),
        ("Zetterberg H.", "Sweden", 1082, 95),
        ("Sahebkar A.", "Iran", 1011, 78),
        ("Larijani B.", "Iran", 657, 65),
        ("Smith L.", "United Kingdom", 613, 72)
    ]
    
    author_id_counter = 1
    pub_id_counter = 1
    
    for name, country, total_pubs, h_index in top_ha_authors:
        author_id = f"AUTHOR_{author_id_counter}"
        author_id_counter += 1
        
        # Create publications for this author (distributed across years)
        author_publications = []
        yearly_distribution = [200, 250, 240, 235, 249]  # 2020-2024
        
        for year_idx, year in enumerate(range(2020, 2025)):
            year_pubs = yearly_distribution[year_idx]
            
            for pub_idx in range(year_pubs):
                publication = Publication(
                    scopus_id=f"PUB_{pub_id_counter}",
                    title=f"Sample Publication {pub_id_counter}",
                    year=year,
                    journal=np.random.choice(TOP_ORTHOPAEDIC_JOURNALS),
                    authors=[author_id] + [f"COAUTHOR_{np.random.randint(1000, 9999)}" for _ in range(np.random.randint(2, 8))],
                    keywords=[f"keyword{i}" for i in range(np.random.randint(1, 6))],
                    citation_count=np.random.randint(0, 50),
                    document_type="article",
                    source_id=f"SOURCE_{np.random.randint(1, 20)}"
                )
                
                author_publications.append(publication)
                sample_publications.append(publication)
                pub_id_counter += 1
        
        # Create author object
        author = Author(
            scopus_id=author_id,
            name=name,
            affiliation="Sample University",
            country=country,
            h_index=h_index,
            total_citations=total_pubs * 15,  # Rough estimate
            publications=author_publications
        )
        
        sample_authors.append(author)
    
    return sample_publications, sample_authors

if __name__ == "__main__":
    # Test with sample data
    print("Creating sample data for testing...")
    publications, authors = create_sample_data()
    
    print(f"Created {len(publications)} publications and {len(authors)} authors")
    
    # Test classification
    for author in authors:
        for year in range(2020, 2025):
            classification = author.classify_productivity(year)
            annual_count = author.get_annual_publication_count(year)
            print(f"{author.name} ({year}): {annual_count} papers -> {classification.value}") 