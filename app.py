import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import tool, SerperDevTool, ScrapeWebsiteTool
import json
from typing import Dict, List
# from io import BytesIO
import os
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI 
from spider import Spider
import sys
import re
import toml

secrets = toml.load("secrets.toml")
all_data_txt = ''

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
        'proxy_enabled': True,
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

class CompanyViewer:
    def __init__(self, data):
        """
        Initialize the CompanyViewer with a data dictionary.
        
        :param data: Dictionary containing company information.
        """
        self.companies = data
        if 'company_data' not in st.session_state:
            st.session_state.company_data = None
            st.session_state.current_page = 1

    @staticmethod
    def _format_attribute_name(attr):
        """
        Format attribute names by capitalizing each word and removing underscores.
        
        :param attr: Attribute name as a string.
        :return: Formatted attribute name.
        """
        words = attr.split('_')
        formatted_words = [word.capitalize() for word in words]
        return ' '.join(formatted_words)

    @staticmethod
    def _flatten_dict(d, parent_key='', sep='_'):
        """
        Flatten a nested dictionary into a flat dictionary with concatenated keys.
        
        :param d: Dictionary to flatten.
        :param parent_key: Parent key to prepend to the current keys.
        :param sep: Separator for concatenated keys.
        :return: Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(CompanyViewer._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, ', '.join(map(str, v)) if v else 'N/A'))
            else:
                items.append((new_key, str(v) if v is not None else 'N/A'))
        return dict(items)

    def _company_to_dataframe(self, company):
        """
        Convert a company dictionary to a flat key-value DataFrame.
        
        :param company: Dictionary representing a company.
        :return: DataFrame containing flattened company data.
        """
        flattened = self._flatten_dict(company)
        df = pd.DataFrame.from_dict(flattened, orient='index', columns=['Value'])
        df.index = [self._format_attribute_name(idx) for idx in df.index]
        df.index.name = 'Attribute'
        return df.reset_index()

    def render(self):
        """
        Render the Streamlit interface for viewing company data.
        """
        # Initialize session states
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1

        # Company selection
        selected_company_name = st.selectbox(
            "Select a Company",
            [company['Company name'] for company in self.companies]
        )

        # Find selected company
        selected_company = next(
            (company for company in self.companies if company['Company name'] == selected_company_name),
            None
        )

        # Update session state if company changes
        if selected_company != st.session_state.get('company_data'):
            st.session_state.company_data = selected_company
            st.session_state.current_page = 1

        if selected_company:
            df = self._company_to_dataframe(selected_company)

            # Pagination logic
            items_per_page = 10
            total_items = len(df)
            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

            # Define callback functions for buttons
            def handle_prev():
                st.session_state.current_page -= 1
                
            def handle_next():
                st.session_state.current_page += 1
                
            def handle_page(page):
                st.session_state.current_page = page

            # Create a container for pagination
            pagination_container = st.container()
            
            with pagination_container:
                cols = st.columns([1] * (total_pages + 2), gap="small")

                # Previous button
                with cols[0]:
                    st.button(
                        "❮",
                        key="prev",
                        on_click=handle_prev,
                        disabled=st.session_state.current_page <= 1,
                        use_container_width=True
                    )

                # Page number buttons
                for page_num in range(1, total_pages + 1):
                    with cols[page_num]:
                        st.button(
                            str(page_num),
                            key=f"page_{page_num}",
                            on_click=handle_page,
                            args=(page_num,),
                            type="primary" if page_num == st.session_state.current_page else "secondary",
                            use_container_width=True
                        )

                # Next button
                with cols[-1]:
                    st.button(
                        "❯",
                        key="next",
                        on_click=handle_next,
                        disabled=st.session_state.current_page >= total_pages,
                        use_container_width=True
                    )

            # Display the current page data
            start_idx = (st.session_state.current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, total_items)

            st.dataframe(
                df.iloc[start_idx:end_idx],
                hide_index=True,
                use_container_width=True,
            )

class BusinessIntelligenceScraper:
    def __init__(self, model_name: str, input_way: str):
        self.model_name = model_name
        self.input_way = input_way
        if model_name == "Groq":
            self.llm_api_calls = 5
        elif model_name == "Gemini":
            self.llm_api_calls = 15
        else:
            self.llm_api_calls = None

    def create_agents(self):
        llm = create_llm(self.model_name)

        crawler_agent = Agent(
            role='Web Crawler',
            goal='Fetch webpage content accurately and efficiently',
            backstory='''I am a specialized web crawler that fetches content from webpages.
            I ensure the content is properly extracted and formatted for analysis.''',
            llm=llm,
            max_iter=8,
            max_rpm=self.llm_api_calls,
            max_execution_time=60*10,
            tools=[crawl_webpage],
            verbose=True
        )
        researcher = Agent(
            role='Corporate Research Expert',
            goal=st.session_state.agent_goal,
            backstory="You are an AI skilled in corporate intelligence, capable of extracting detailed information from multiple data sources, with a focus on providing accurate and organized insights for business analysis. You're especially good at making reasonable estimates when direct data isn't available.",
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            max_rpm=self.llm_api_calls,
            verbose=True,
            max_iter=8,
            max_execution_time=60*10,
            llm=llm
        )
        company_research_agent = Agent(
            role="Company Researcher",
            goal="Extract and compile detailed information about a specified company.",
            verbose=True,
            memory=True,
            llm=llm,
            max_rpm=self.llm_api_calls,
            max_iter=8,
            max_execution_time=60*10,
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
                    f"Conduct comprehensive research on the {url} of company website."
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
                    "A structured report containing all the requested details about the only only provided."
                    "The output must be a valid JSON dictionary with keys corresponding to the requested details"
                    "For Example:"
                    '''{
                        "Company name": "Example Corp",
                        "Website": "https://example.com",
                        "Year established": 1998,
                        "Estimated employees": "100-500",
                        "Estimated annual revenue": "$10M-$50M",
                        "General company email": "info@example.com",
                        "Additional company email": "press@example.com",
                        "Phone number 1": "+123456789",
                        "Phone number 2": "+987654321",
                        "Fax Number": "+123456788",
                        "Address from the website": "123 Example Street, City, Country",
                        "Website summary": "Example Corp is a leading provider of innovative solutions.",
                        "Services provided": ["Service 1", "Service 2"],
                        "Official address if different": "456 Official Lane, City, Country",
                        "Map": "https://maps.google.com/example",
                        "Company LinkedIn": "https://linkedin.com/company/example",
                        "Website contact us form link": "https://example.com/contact",
                        "List of all employees": [
                            {"Name": "John Smith", "Title": "CEO", "Email": "john.smith@example.com", "LinkedIn": "https://linkedin.com/in/john"},
                            {"Name": "Watson", "Title": "VP of sales", "Email": "watson@example.com", "LinkedIn": "https://linkedin.com/in/waston"},
                            #Other senior people name, title email, linkedin
                            
                        ],
                        "Twitter/X": {"URL": "https://twitter.com/example", "# of posts": 200, "# of followers": 10000, "Summary": "Tech innovations."},
                        "Instagram": {"URL": "https://instagram.com/example", "# of posts": 150, "# of followers": 8000, "Summary": "Visual highlights."},
                        "Facebook page": {"URL": "https://facebook.com/example", "# of posts": 300},
                        "List of similar companies/Competitors": ["Competitor 1", "Competitor 2"],
                        "Services they need": ["Service A", "Service B"],
                        "Services they use": ["Service C", "Service D"],
                        "List of their current customers": ["Customer 1", "Customer 2"],
                        "News": ["News article 1", "News article 2"],
                        "Press releases": ["Press release 1", "Press release 2"],
                        "Blogs": ["Blog post 1", "Blog post 2"],
                        "Events - virtual": ["Event 1", "Event 2"],
                        "Events - In person": ["Event 3", "Event 4"],
                        "Trade Shows": ["Trade show 1", "Trade show 2"],
                        "Funding, stock market, financials": {"Funding": "$1M", "Stock market": "NASDAQ:EXMP", "Estimated Value": "$100M"},
                        "Revenue": "$50M",
                        "Job postings": [{"Title": "Software Engineer", "Summary": "Develop innovative solutions."}],
                        "EIN": "12-3456789",
                        "SIC Code": "1234",
                        "NAICS Code": "5678",
                        "SIC Description": "Software Development",
                        "Longitude": "12.3456",
                        "Latitude": "78.9012",
                        "Use case 1": "Summarize from website.",
                        "Customer testimonial 1": "Great service!",
                        "Customer review": "Positive reviews from customers.",
                        "Ratings overall": 4.5,
                        "Word cloud": ["innovation", "technology", "solutions"]
                    }'''
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
        global all_data_txt
        # Create crew
        crew = Crew(
            agents=agent_list,
            tasks=task_list,
            process=Process.sequential
        )

        # Execute crew
        result = crew.kickoff()
        all_data_txt += str(result) + '\n'
        result = str(result).replace("null", '"null"')

        try:
            try:
                extracted_info = eval(str(result).replace('```json', '').replace('```', '').strip())
                return extracted_info
            except:
            
                # Parse the analyzer's response as JSON
                if st.session_state.s_c == 'Find Similar Companies':
                    extracted_info = re.findall(r'\{.*?\]\}', str(result).replace('\n', '')) 
                    if extracted_info:
                        extracted_info = eval(extracted_info[0]) 
                        return extracted_info
                else:
                    # st.error("Failed to parse JSON response")
                    return {}
                
        except json.JSONDecodeError as e:
            # st.error(f"Failed to parse JSON response: {str(e)}")
            return {}
        
def create_llm(model_name):
    try:
        if model_name == "llama3.2":
            return ChatOpenAI(
                model="ollama/llama3.2",
                base_url="http://localhost:11434"
            )
        
        if model_name == "Groq":
            return LLM(model="groq/llama-3.2-1b-preview")
        
        if model_name == "Gemini":
            return LLM(
                model="gemini/gemini-1.5-flash",
                api_key=st.session_state.gemini_api_key
            )
        
        # Default case: gpt-4o-mini
        return ChatOpenAI(model="gpt-4o-mini")
    except Exception as e:
        st.error(f"An unexpected error occurred while creating the LLM: {e}")
        return None 

def similar_comapnies_url_find(url, model_name):
    llm = create_llm(model_name)
    if model_name == 'Groq':
        max_llm_api_calls = 5
    elif model_name == "Gemini":
        max_llm_api_calls = 15
    else:
        max_llm_api_calls = None
    
    company_analysis_agent = Agent(
        role="Company Research Analyst",
        goal="Research the given company and identify similar companies.",
        verbose=True,
        memory=True,
        llm=llm,
        max_rpm=max_llm_api_calls,
        max_iter=8,
        max_execution_time=60*10,
        backstory=(
            "An expert in company research, skilled in uncovering detailed insights about businesses "
            "and identifying similar companies in the market."
        ),
        tools=[SerperDevTool(), ScrapeWebsiteTool()]
    )
    
    # Task 1: Research the company
    company_research_task = Task(
        description=(
            f"Research the company at {url} and provide a detailed analysis. "
            "Identify the industry, key services/products, target audience, and other distinguishing features. "
            "Your output should be a JSON object in the following format:\n\n"
            "{\n"
            "  \"industry\": \"<industry>\",\n"
            "  \"services\": [\"service1\", \"service2\"],\n"
            "  \"key_features\": [\"feature1\", \"feature2\"]\n"
            "}"
        ),
        expected_output=(
            "A JSON dictionary containing the company's industry, services, and key features."
        ),
        agent=company_analysis_agent,
    )
    
    # Task 2: Find similar companies
    similar_companies_task = Task(
        description=(
            "Using the analysis from the previous task, identify companies that are "
            f"similar in terms of industry, size, or offerings. Compile a list  of {st.session_state.comp_num} companies of their website URLs. "
            "Your output should be a JSON object in the following format:\n\n"
            "{\n"
            "  \"similar_companies\": [\n"
            "    \"https://example1.com\",\n"
            "    \"https://example2.com\",\n"
            "  ]\n"
            "}"
        ),
        expected_output=(
            "A JSON dictionary containing a list of URLs for similar companies."
        ),
        agent=company_analysis_agent,
    )
    
    # Create crew
    crew = Crew(
        agents=[company_analysis_agent],
        tasks=[company_research_task, similar_companies_task],
        process=Process.sequential
    )
    
    # Execute crew
    result = crew.kickoff()
    
    try:
        # Parse the analyzer's response as JSON
        extracted_info = re.findall(r'"https?://[^"]+"', str(result)) 
        if extracted_info:
            extracted_info = [url.strip('"') for url in extracted_info]
            
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {str(e)}")

    return extracted_info

def save_and_provide_download_link(results):
    try:
        filename = "Result_data.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str(results))
        
        with open(filename, "rb") as f:
            st.download_button(
                label="Download Results",
                data=f,
                file_name=filename,
                mime="text/plain"
            )
        # if os.path.exists(filename):
        #     os.remove(filename)
    except IOError as e:
        st.error(f"File I/O error: {e}")
    except Exception as e:
        st.error(f"Failed to save and provide download link: {e}")

def process_urls(urls: List[str],
                 model_name: str,
                 input_way_data, 
                 my_bar
                 ) -> pd.DataFrame:
    """Process multiple URLs and return results as a DataFrame."""
    input_urls = urls
    scraper = BusinessIntelligenceScraper(model_name, input_way_data)
    if input_way_data == 'Find Similar Companies':
        for url in urls:
            similar_comp = similar_comapnies_url_find(url, model_name)
            input_urls = similar_comp
            st.success(f'links: {input_urls}')
            if not urls:
                st.warning('No similar company find')
    my_bar.progress(10)
    results = []
    total_urls = len(input_urls)
    p_num = int(90/total_urls)
    bb = 10
    for url in (input_urls):
        bb += p_num
        try:
            info = scraper.process_url(url)
            if info:
                results.append(info)
        
        except Exception as e:
            # st.error(f"Error processing URL {url}: {e}")
            pass
        my_bar.progress(bb)
    if results:
        with open('results.json', 'w') as f:
            json.dump(results, f)
    return pd.DataFrame(results)

def main():
    st.title("Business Intelligence Extractor")
    st.write("Upload an Excel file with URLs to extract business intelligence.")

    # Model selection switch
    model_name = st.radio(
        "Select AI Model",
        options=["Gemini", "llama3.2", "Groq", "gpt-4o-mini"],
        index=0,
        help="Choose between gpt-4o-mini (default) and llama3.2 from Ollama."
    )
    if model_name == "gpt-4o-mini":
        open_ai_api_key = st.text_input("OpenAI API Key", type="password")
        if open_ai_api_key:
            os.environ["OPENAI_API_KEY"] = open_ai_api_key
        else:
            st.error("Please provide an OpenAI API key.")
    if model_name == "Gemini":
        st.session_state.gemini_api_key = secrets['GEMINI_API_KEY']
        # gemini_api_key = st.text_input("Gemini API Key", type="password")
        # if gemini_api_key:
        #     st.session_state.gemini_api_key = secrets['GEMINI_API_KEY']
        # else:
        #     st.error("Please provide a Gemini API key.")
            
    os.environ['GROQ_API_KEY'] = secrets['GROQ_API_KEY']
    # if model_name == "Groq":
    #     groq_api_key = st.text_input("Groq API Key", type="password")
    #     if groq_api_key:
    #         st.session_state.groq_api_key = groq_api_key
    #     else:
    #         st.error("Please provide a Groq API key.")

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
    similar_company = None
    st.session_state.agent_goal = st.text_area("Edit agent goal as you want", value='''Conduct an in-depth analysis of company to extract key financials, employee details, tech stack, services, competitors, and other relevant information. If direct data is unavailable, provide educated estimates based on industry standards and similar companies.''')
    st.write("Select any two options, but one must be from the first two.")
    excel_file_selected = st.checkbox("Excel File")
    give_input_selected = st.checkbox("Give Input")
    find_similar_selected = st.checkbox("Find Similar Companies")
    if find_similar_selected is True:
        st.session_state.s_c = find_similar_selected
    else:
        st.session_state.s_c = None
    
    if not excel_file_selected and not give_input_selected:
        st.error("You must select at least one option from 'Excel File' or 'Give Input'.")
    elif excel_file_selected and give_input_selected:
        st.error("You must select one 'Excel File' or 'Give Input'")
    else:
        if excel_file_selected:
            uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
        if give_input_selected:
            urls = st.text_area("Enter URLs (one per line)", value="https://www.example.com\nhttps://www.example.com")            
        if find_similar_selected is True:
            similar_company = 'Find Similar Companies'
    st.session_state.comp_num = st.number_input("Number of similar companies to extract", min_value=2, max_value=10, value=2)

    if excel_file_selected and uploaded_file is not None:
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
            
    elif give_input_selected and urls is not None:
        urls = [url for url in urls.split('\n') if url.strip()]
    
            
    if st.button("Extract Business Intelligence"):
        
        progress_bar = st.progress(2)
        process_output_expander = st.expander("Processing Output:")
        sys.stdout = StreamToExpander(process_output_expander)
        try:
            results_df = process_urls(urls, model_name, similar_company, progress_bar)
            if all_data_txt:
                save_and_provide_download_link(all_data_txt)
            
                
            # output = BytesIO()
            # results_df.to_excel(output, index=False, engine='openpyxl')
            # output.seek(0)

            # # Create download button
            # st.download_button(
            #     label="Download Results",
            #     data=output,
            #     file_name="business_intelligence_results.xlsx",
            #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            # )

            # Display results
            # st.write("Extracted Business Intelligence:")
            # if find_similar_selected is not True:
            #     st.dataframe(results_df)
            
        except Exception as e:
            # st.error(f"An error occurred: {str(e)}")
            pass  
    ot_results = None
    if os.path.exists("results.json"):
        with open("results.json", "r", encoding="utf-8") as f:
            ot_results = json.load(f)
        if ot_results:
            
            if similar_company == 'Find Similar Companies':
                try:
                    viewer = CompanyViewer(ot_results)
                    viewer.render()
                except Exception as e:
                    # st.error(f"Error rendering company viewer: {e}")
                    pass
            
if __name__ == "__main__":
    main()