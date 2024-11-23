import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
import json
from typing import Dict, List
import tempfile
import os
import toml
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI as LLM
from spider import Spider
# Load secrets
try:
    secrets = toml.load("D:\\finalproject\\ai\\secrets.toml")
    os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY1"]
    os.environ["SPIDER_API_KEY"] = secrets["SPIDER_API_KEY"]
except KeyError as e:
    st.error(f"Missing key in secrets.toml: {str(e)}")
except FileNotFoundError:
    st.error("secrets.toml file not found. Ensure the file is present in the working directory.")

# Utility function to validate URLs
def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


@tool("Web Content Crawler")
def crawl_webpage(url: str) -> str:
    """
    Crawls a webpage and extracts its content.
    
    Args:
        url (str): The URL of the webpage to crawl
        
    Returns:
        str: The content of the webpage
    """
    spider = Spider()
    crawl_params = {
        'limit': 1,
        'fetch_page_content': True,
        'metadata': False,
        'return_format': 'markdown'
    }
    try:
        crawl_result = spider.crawl_url(url, params=crawl_params)
        return crawl_result[0]['content']
    except Exception as e:
        return f"Error crawling {url}: {str(e)}"
        
# Business Intelligence Tool with Model Switching
class BusinessIntelligenceScraper:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def create_llm(self):
        if self.model_name == "llama3.2":
            return LLM(
                model="ollama/llama3.2",
                base_url="http://localhost:11434"
            )
        return LLM(model="gpt-4o-mini")

    def create_agents(self):
        llm = self.create_llm()

        # Crawler Agent
        crawler_agent = Agent(
            role='Web Crawler',
            goal='Scrape and crawl web content for company analysis.',
            backstory='An expert web crawler with access to detailed web scraping tools.',
            llm=llm,
            tools=[crawl_webpage],
            verbose=True
        )

        # Analyzer Agent
        analyzer_agent = Agent(
            role='Data Analyzer',
            goal='Analyze scraped data to extract financials, employee info, and other insights.',
            backstory='A data analyst skilled at processing web content into actionable insights.',
            llm=llm,
            verbose=True
        )

        return crawler_agent, analyzer_agent

    def process_url(self, url: str) -> Dict:
        crawler_agent, analyzer_agent = self.create_agents()

        # Crawling Task
        crawl_task = Task(
            description=f"Crawl and scrape the webpage at {url} using SpiderTool.",
            expected_output="HTML content of the webpage in markdown format.",
            agent=crawler_agent
        )

        # Analyzing Task
        analyze_task = Task(
            description="Analyze scraped content to extract financial data, employee info, services, and competition.",
            expected_output='''JSON string containing keys: 'financials', 'employees', 'services', 'competition', and 'government_info'.''',
            agent=analyzer_agent,
            context=[crawl_task] 
        )

        # Create crew
        crew = Crew(
            agents=[crawler_agent, analyzer_agent],
            tasks=[crawl_task, analyze_task],
            process=Process.sequential
        )

        # Execute crew
        result = crew.kickoff(inputs={"url": url})

        try:
            # Parse the analyzer's response as JSON
            extracted_info = json.loads(str(result).replace('```json', '').replace('```', '')) 
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response: {str(e)}")
            extracted_info = {
                'financials': '',
                'employees': '',
                'services': '',
                'competition': '',
                'government_info': ''
            }

        return extracted_info

def process_urls(urls: List[str], model_name: str) -> pd.DataFrame:
    """Process multiple URLs and return results as a DataFrame."""
    scraper = BusinessIntelligenceScraper(model_name)
    results = []

    for url in urls:
        info = scraper.process_url(url)
        info['url'] = url
        results.append(info)

    return pd.DataFrame(results)

# Streamlit UI
def main():
    st.title("Business Intelligence Extractor")
    st.write("Upload an Excel file with URLs to extract business intelligence.")

    # Model selection switch
    model_name = st.radio(
        "Select AI Model",
        options=["gpt-4o-mini", "llama3.2"],
        index=0,
        help="Choose between gpt-4o-mini (default) and llama3.2 from Ollama."
    )

    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'url' not in df.columns:
                st.error("The Excel file must contain a column named 'url'.")
                return

            urls = df['url'].tolist()

            # Validate URLs
            invalid_urls = [url for url in urls if not is_valid_url(url)]
            if invalid_urls:
                st.error(f"Invalid URLs detected: {', '.join(invalid_urls)}")
                return

            if st.button("Extract Business Intelligence"):
                with st.spinner('Processing URLs...'):
                    try:
                        results_df = process_urls(urls, model_name)

                        # Create temporary file for download
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                            results_df.to_excel(tmp.name, index=False)

                            # Create download button
                            with open(tmp.name, 'rb') as f:
                                st.download_button(
                                    label="Download Results",
                                    data=f,
                                    file_name="business_intelligence_results.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                            # Clean up temporary file
                            os.unlink(tmp.name)

                        # Display results
                        st.write("Extracted Business Intelligence:")
                        st.dataframe(results_df)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")

if __name__ == "__main__":
    main()
