import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool, SerperDevTool, ScrapeWebsiteTool
import json
from typing import Dict, List
from io import BytesIO
import os
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI as LLM
from spider import Spider
import sys
import re

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
      
class StreamToExpander:
    def __init__(self, expander, buffer_limit=10000):
        self.expander = expander
        self.buffer = []
        self.buffer_limit = buffer_limit

    def write(self, data):
        # Clean ANSI escape codes from output
        cleaned_data = re.sub(r'\x1B\[\d+;?\d*m', '', data)
        if len(self.buffer) >= self.buffer_limit:
            self.buffer.pop(0)
        self.buffer.append(cleaned_data)

        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()

    def flush(self):
        if self.buffer:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()      
   
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

        crawler_agent = Agent(
            role='Web Crawler',
            goal='Fetch webpage content accurately and efficiently',
            backstory='''I am a specialized web crawler that fetches content from webpages.
            I ensure the content is properly extracted and formatted for analysis.''',
            llm=llm,
            tools=[crawl_webpage],
            verbose=True
        )
        researcher = Agent(
            role='Corporate Research Expert',
            goal=f"Conduct an in-depth analysis of company to extract key financials, employee details, tech stack, services, competitors, and other relevant information. If direct data is unavailable, provide educated estimates based on industry standards and similar companies.",
            backstory="You are an AI skilled in corporate intelligence, capable of extracting detailed information from multiple data sources, with a focus on providing accurate and organized insights for business analysis. You're especially good at making reasonable estimates when direct data isn't available.",
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True,
            llm=llm
        )

        return crawler_agent, researcher

    def process_url(self, url: str) -> Dict:
        crawler_agent, researcher = self.create_agents()

        # Crawling Task
        crawler_task = Task(
            description=f'''Analyze the webpage content and extract of {url}:
            1. Name of person
            2. Phone number
            3. Email address
            4. Social media contacts
            
            Return the information in a valid JSON dictionary format with keys:
            'name', 'phone', 'email', 'social_media'
            ''',
            expected_output='''A JSON string containing the following keys:
            {
                "name": "extracted name or empty string",
                "phone": "extracted phone number or empty string",
                "email": "extracted email or empty string",
                "social_media": "extracted social media links or empty string"
            }''',
            agent=crawler_agent, 
            verbose = True
        )

        researcher_task = Task(
            description=f"""Research {url} and answer the following questions:

            1. Show me financial data for this company:

            2. Who works at this company?

            3. What type of software does this company use?

            4. What are all the services they offer?

            5. Who used to work at this company?

            6. Are there any blogs or articles related to this company?

            7. Find all government information about this company:

            8. Who are their competitors or similar companies?

            9. For each type of company, save relevant government websites:

            10. The agent should remember the types of questions and prompts to ask based on the business category, example questions for plumbing companies will be different from ones for finding all supermarkets in a certain city that have organic foods and wheelchair access:

            11. If no direct financials are available, can you find old financial data or similar companies to make an estimate?

            12. Can the AI estimate annual revenue and employee count when information is limited?
            """,
                        agent=researcher,
                        expected_output="""
            A JSON dictionary containing the following keys:
            {
                "name": "extracted name or empty string",
                "phone": "extracted phone number or empty string",
                "email": "extracted email or empty string",
                "social_media": "extracted social media links or empty string"
                "report": "A structured report answering each question as follows:
                        1. Financial Data
                        2. Current Employees
                        3. Software Used
                        4. Services and Products Offered
                        5. Former Employees
                        6. Media Presence
                        7. Government Information
                        8. Competitors and Similar Companies
                        9. Reference Information for Government Sites
                        10. Business Category-Specific Prompts
                        11. Financial Estimates
                        12. Revenue and Employee Estimates

                        Each section should contain answers or reasoned estimates based on industry standards where direct data is unavailable. Make sure this report is in report format."
            }"""
                    )
        # Create crew
        crew = Crew(
            agents=[crawler_agent, researcher],
            tasks=[crawler_task, researcher_task],
            process=Process.sequential
        )

        # Execute crew
        result = crew.kickoff()

        try:
            # Parse the analyzer's response as JSON
            extracted_info = json.loads(str(result).replace('```json', '').replace('```', '')) 
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response: {str(e)}")

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
    if model_name == "gpt-4o-mini":
        open_ai_api_key = st.text_input("OpenAI API Key", type="password")
        if open_ai_api_key:
            os.environ["OPENAI_API_KEY"] = open_ai_api_key
        else:
            st.error("Please provide an OpenAI API key.")
    spider_api_key = st.text_input("Spider Crawl API Key", type="password")
    if spider_api_key:
        os.environ["SPIDER_API_KEY"] = spider_api_key
    else:
        st.error("Please provide a Spider Crawl API key.")
    
    serper_api_key = st.text_input("Serper API Key", type="password")
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
    else:
        st.error("Please provide a Serper API key")
        
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
                    process_output_expander = st.expander("Processing Output:")
                    sys.stdout = StreamToExpander(process_output_expander)
                    try:
                        results_df = process_urls(urls, model_name)

                        output = BytesIO()
                        results_df.to_excel(output, index=False, engine='openpyxl')
                        output.seek(0)

                        # Create download button
                        st.download_button(
                            label="Download Results",
                            data=output,
                            file_name="business_intelligence_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        # Display results
                        st.write("Extracted Business Intelligence:")
                        st.dataframe(results_df)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")


if __name__ == "__main__":
    main()