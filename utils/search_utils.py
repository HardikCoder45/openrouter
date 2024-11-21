from duckduckgo_search import DDGS
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import time
import re

class SearchEngine:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.ddgs = DDGS()

    def clean_query(self, query: str) -> str:
        """Clean and enhance the search query."""
        # Remove special characters but keep spaces
        query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
        # Remove extra spaces
        query = ' '.join(query.split())
        return query

    def search(self, query: str) -> List[Dict]:
        """Perform a DuckDuckGo search and return results."""
        try:
            cleaned_query = self.clean_query(query)
            results = []
            
            # Try different search strategies
            search_strategies = [
                (f"{cleaned_query} programming tutorial example", "y"),  # Recent results
                (f"{cleaned_query} code documentation github", "y"),     # GitHub results
                (f"{cleaned_query} stackoverflow solution", "y"),        # StackOverflow results
                (cleaned_query, None)                                    # General results
            ]
            
            for search_query, timelimit in search_strategies:
                if len(results) < self.max_results:
                    try:
                        search_params = {
                            "keywords": search_query,
                            "max_results": self.max_results - len(results),
                            "region": "wt-wt",
                            "safesearch": "off"
                        }
                        if timelimit:
                            search_params["timelimit"] = timelimit
                            
                        new_results = list(self.ddgs.text(**search_params))
                        
                        # Filter out duplicate results
                        for result in new_results:
                            if not any(r['link'] == result['link'] for r in results):
                                results.append(result)
                                
                        if results:
                            break  # Stop if we found results
                            
                    except Exception as e:
                        st.error(f"Search strategy error: {str(e)}")
                        continue
            
            # Process and clean results
            processed_results = []
            for r in results:
                # Clean and format the content
                snippet = r['body'].replace('\n', ' ').strip()
                # Remove very short or irrelevant snippets
                if len(snippet) > 50:  # Minimum length threshold
                    processed_results.append({
                        'title': r['title'],
                        'link': r['link'],
                        'snippet': snippet
                    })
            
            return processed_results

        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

    def format_search_results(self, results: List[Dict]) -> str:
        """Format search results for the LLM context."""
        if not results:
            return "No relevant search results found."
        
        formatted = "🔎 **Relevant Information Found:**\n\n"
        for i, result in enumerate(results, 1):
            # Format title
            formatted += f"### Source {i}: {result['title']}\n"
            
            # Format snippet with proper code detection
            snippet = result['snippet']
            if any(code_indicator in snippet.lower() for code_indicator in 
                  ['code', 'function', 'class', 'def ', 'var ', 'const ', 'let ']):
                formatted += "```\n" + snippet + "\n```\n"
            else:
                formatted += f"{snippet}\n"
            
            # Add source link
            formatted += f"📚 *[Reference Link]({result['link']})*\n\n"
            formatted += "---\n\n"
        
        return formatted

    async def async_search(self, query: str) -> List[Dict]:
        """Asynchronous search method with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(self.executor, self.search, query)
                if results:
                    return results
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Search failed after {max_retries} attempts: {str(e)}")
                    return []
        return []