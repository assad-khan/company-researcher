import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import tool, SerperDevTool, ScrapeWebsiteTool
import json
from typing import Dict, List
from io import BytesIO
import os
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI 
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
    def __init__(self, model_name: str, input_way: str):
        self.model_name = model_name
        self.input_way = input_way

    def create_agents(self):
        llm = create_llm(self.model_name)

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
            goal=st.session_state.agent_goal,
            backstory="You are an AI skilled in corporate intelligence, capable of extracting detailed information from multiple data sources, with a focus on providing accurate and organized insights for business analysis. You're especially good at making reasonable estimates when direct data isn't available.",
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True,
            llm=llm
        )
        company_research_agent = Agent(
            role="Company Researcher",
            goal="Extract and compile detailed information about a specified company.",
            verbose=True,
            memory=True,
            llm=llm,
            tools=[SerperDevTool(), crawl_webpage],
            backstory=(
                "An expert in online data aggregation and analysis, "
                "dedicated to providing comprehensive company insights."
            ),
        )
        if self.input_way == "Find Similar Companies":
            return company_research_agent
        else:
            return crawler_agent, researcher

    def create_tasks(self, url: str):
        if self.input_way == "Find Similar Companies":
            company_resecher_agent = self.create_agents()
            company_research_task = Task(
                description=(
                    f"Conduct comprehensive research on the {url}. "
                    "Gather the following details:\n"
                    "- Company name\n"
                    "- Website\n"
                    "- Year established\n"
                    "- Estimated employees\n"
                    "- Estimated annual revenue\n"
                    "- General company email\n"
                    "- Additional company email\n"
                    "- Phone numbers\n"
                    "- Fax number\n"
                    "- Website address\n"
                    "- Website summary\n"
                    "- Services provided\n"
                    "- Official address\n"
                    "- Map details\n"
                    "- Company LinkedIn\n"
                    "- Contact form link\n"
                    "- Employees list\n"
                    "- Social media details (Twitter, Instagram, Facebook)\n"
                    "- Competitors\n"
                    "- Predicted services they need\n"
                    "- Predicted services they use\n"
                    "- Current customers\n"
                    "- News, blogs, press releases\n"
                    "- Events (virtual, in-person, trade shows)\n"
                    "- Financial details (funding, stock market data, revenue)\n"
                    "- Job postings\n"
                    "- EIN, SIC Code, NAICS Code, SIC Description\n"
                    "- Geolocation (Longitude, Latitude)\n"
                    "- Use cases (summarized from website)\n"
                    "- Customer testimonials\n"
                    "- Customer reviews and ratings\n"
                    "- Word cloud (25-50 words)\n"
                ),
                expected_output=(
                    "A structured report containing all the requested details about the company. "
                    "The output must be well-organized and include all the specified data points. Also output must be in a valid JSON format."
                ),
                agent=company_resecher_agent,
            )
            
            return [company_resecher_agent], [company_research_task]
        else:
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
            
            return [crawler_agent, researcher], [crawler_task, researcher_task] 
            
    def process_url(self, url: str) -> Dict:
        agent_list, task_list = self.create_tasks(url)
        
        # Create crew
        crew = Crew(
            agents=agent_list,
            tasks=task_list,
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

def create_llm(model_name):
        if model_name == "llama3.2":
            return ChatOpenAI(
                model="ollama/llama3.2",
                base_url="http://localhost:11434"
            )
        
        if model_name == "Groq":
            return LLM(model="groq/llama-3.2-1b-preview", 
                       api_key=st.session_state.groq_api_key)
        
        return ChatOpenAI(model="gpt-4o-mini")

def similar_comapnies_url_find(url, model_name):
    llm = create_llm(model_name)
    company_analysis_agent = Agent(
        role="Company Similarity Analyst",
        goal="Analyze the given company URL and identify similar companies.",
        verbose=True,
        memory=True,
        llm=llm,
        backstory=(
            "A specialist in company analysis and market research, "
            "skilled in identifying competitors and similar businesses."
        ),
        tools=[SerperDevTool(), ScrapeWebsiteTool()]
    )
    
    similar_companies_task = Task(
        description=(
            f"Analyze the company URL {url} and identify companies that are "
            "similar in terms of industry, size, or offerings. Use available online resources "
            "to compile a list of three similar companies' URLs. "
            "Your output should be a JSON object in the following format:\n\n"
            "{\n"
            "  \"similar_companies\": [\n"
            "    \"https://example1.com\",\n"
            "    \"https://example2.com\",\n"
            "    \"https://example3.com\"\n"
            "  ]\n"
            "}"
        ),
        expected_output=(
            "A JSON dictionary containing a list  of URLs for companies similar to the given company."
        ),
        agent=company_analysis_agent,
    )

    # Create crew
    crew = Crew(
        agents=[company_analysis_agent],
        tasks=[similar_companies_task],
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
    
def process_urls(urls: List[str], model_name: str, input_way_data) -> pd.DataFrame:
    """Process multiple URLs and return results as a DataFrame."""
    scraper = BusinessIntelligenceScraper(model_name, input_way_data)
    if input_way_data == 'Find Similar Companies':
        url_list = []
        for url in urls:
            similar_comp = similar_comapnies_url_find(url, model_name)
            urlss = similar_comp.get('similar_companies', [])
            url_list.append(urlss)
            st.success(f'links: {urlss}')
            if not urls:
                st.warning('No similar company find')
        urls = url_list
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
        options=["gpt-4o-mini", "llama3.2", "Groq"],
        index=0,
        help="Choose between gpt-4o-mini (default) and llama3.2 from Ollama."
    )
    if model_name == "gpt-4o-mini":
        open_ai_api_key = st.text_input("OpenAI API Key", type="password")
        if open_ai_api_key:
            os.environ["OPENAI_API_KEY"] = open_ai_api_key
        else:
            st.error("Please provide an OpenAI API key.")
            
    if model_name == "Groq":
        groq_api_key = st.text_input("Groq API Key", type="password")
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key
        else:
            st.error("Please provide a Groq API key.")

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
    st.session_state.agent_goal = st.text_area("Edit agent goal as you want", value='''Conduct an in-depth analysis of company to extract key financials, employee details, tech stack, services, competitors, and other relevant information. If direct data is unavailable, provide educated estimates based on industry standards and similar companies.''')
    input_url_way = st.radio(
        "Select input url way",
        options=["Excel File", "Give Input"],
        index=0,
        help="Choose between upload excel file or give input"
    )
    similar_company  = st.radio(
        "Do you want to find similar companies?",
        options=["Find Similar Companies", "Don't Find Similar Companies"],
        index=0,
        help="Choose between find similar companies or don't find similar companies"
    )
    
    if input_url_way == 'Excel File':
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
    else:
        urls = st.text_area("Enter URLs (one per line)", value="https://www.example.com\nhttps://www.example.com")
    
    
    if input_url_way == 'Excel File' and uploaded_file is not None:
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
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")    
            
    elif input_url_way == 'Give Input' and urls is not None:
        urls = [url for url in urls.split('\n') if url.strip()]
    
            
    if st.button("Extract Business Intelligence"):
        with st.spinner('Processing URLs...'):
            process_output_expander = st.expander("Processing Output:")
            sys.stdout = StreamToExpander(process_output_expander)
            try:
                results_df = process_urls(urls, model_name, similar_company)

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


if __name__ == "__main__":
    main()