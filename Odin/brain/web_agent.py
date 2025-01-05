import os
import logging
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .memory_handler import MemoryHandler
from gensim.utils import simple_preprocess
import sqlite3
import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Thread lock for printing
print_lock = threading.Lock()

class WebAgent:
    def __init__(self, conversation_manager):
        """
        Initialize the WebAgent with a reference to the ConversationManager.
        """
        self.conversation_manager = conversation_manager
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.conversation_manager.OPEN_ROUTER_API_KEY,
        )
        self.memory_handler = MemoryHandler(self.conversation_manager.memory_dir)

    def is_url(self, text):
        """
        Check if the input text contains a URL.
        """
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return bool(url_pattern.search(text))

    def extract_urls(self, text):
        """
        Extract all URLs from the input text.
        """
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return url_pattern.findall(text)

    def scrape_url(self, url):
        """
        Scrape the content of a given URL using Selenium.
        """
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            driver.get(url)
            time.sleep(5)  # Allow time for the page to load
            seo_data = self.scrape_seo_info(driver.page_source, url)
            driver.quit()
            return seo_data
        except (TimeoutException, WebDriverException) as e:
            logging.error(f"Error scraping URL {url}: {str(e)}")
            return None

    def scrape_seo_info(self, page_source, url):
        """
        Extract SEO information from the scraped page source.
        """
        try:
            soup = BeautifulSoup(page_source, 'html.parser')
            seo_data = {
                'title': soup.title.string if soup.title else "No Title",
                'meta_description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else "No Meta Description",
                'h1': [h1.text for h1 in soup.find_all('h1')],
                'h2': [h2.text for h2 in soup.find_all('h2')],
                'h3': [h3.text for h3 in soup.find_all('h3')],
                'h4': [h4.text for h4 in soup.find_all('h4')],
                'h5': [h5.text for h5 in soup.find_all('h5')],
                'h6': [h6.text for h6 in soup.find_all('h6')],
                'paragraphs': [p.text for p in soup.find_all('p')],
                'unordered_lists': [ul.text for ul in soup.find_all('ul')],
                'ordered_lists': [ol.text for ol in soup.find_all('ol')],
                'image_alt_texts': [img['alt'] for img in soup.find_all('img') if img.get('alt')],
                'internal_links': [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('/') or url in a['href']],
                'external_links': [a['href'] for a in soup.find_all('a', href=True) if not a['href'].startswith('/') and url not in a['href']],
                'canonical': soup.find('link', rel='canonical')['href'] if soup.find('link', rel='canonical') else "No Canonical URL",
                'robots': soup.find('meta', attrs={'name': 'robots'})['content'] if soup.find('meta', attrs={'name': 'robots'}) else "No Robots Meta",
                'word_count': len(soup.get_text().split())
            }

            # Organize content by H2 sections
            h2_sections = []
            current_h2 = None
            current_content = []
            for element in soup.find_all(['h2', 'p', 'ul', 'ol']):
                if element.name == 'h2':
                    if current_h2:
                        h2_sections.append({
                            'header': current_h2,
                            'content': "\n".join(current_content)
                        })
                        current_content = []
                    current_h2 = element.text.strip()
                elif element.name in ['p', 'ul', 'ol']:
                    current_content.append(element.text.strip())
            if current_h2:
                h2_sections.append({
                    'header': current_h2,
                    'content': "\n".join(current_content)
                })

            # Generate embeddings for each section
            embeddings = []
            for section in h2_sections:
                embedding = self.memory_handler.sentence_to_vec(section['content'])
                embeddings.append(embedding)

            # Update the Word2Vec model with new sentences
            sentences = [simple_preprocess(section['content']) for section in h2_sections]
            self.memory_handler.update_word2vec_model(sentences)

            return {
                'seo_data': seo_data,
                'h2_sections': h2_sections,
                'embeddings': embeddings
            }
        except Exception as e:
            logging.error(f"Error scraping SEO info: {str(e)}")
            return None

    def save_seo_data(self, url, data):
        """
        Save the scraped SEO data to the database in the `file_chunks` table.
        """
        markdown_content = self.convert_to_markdown(data)
        embedding = self.memory_handler.sentence_to_vec(markdown_content)
        conn = sqlite3.connect(self.conversation_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO file_chunks (timestamp, chunk, embedding, file_name)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            markdown_content,
            json.dumps(embedding.tolist()) if embedding is not None else None,
            url  # Use the URL as the file_name for web content
        ))
        conn.commit()
        conn.close()
        logging.info(f"SEO data and raw content saved to the database for URL: {url}")

    def convert_to_markdown(self, data):
        """
        Convert scraped SEO data into Markdown format.
        """
        markdown_content = []
        seo_data = data.get("seo_data", {})
        h2_sections = data.get("h2_sections", [])
        if seo_data.get("title"):
            markdown_content.append(f"# {seo_data['title']}\n")
        if seo_data.get("meta_description"):
            markdown_content.append(f"**Meta Description:** {seo_data['meta_description']}\n")
        if h2_sections:
            for section in h2_sections:
                markdown_content.append(f"## {section['header']}\n")
                markdown_content.append(f"{section['content']}\n")
        return "\n".join(markdown_content)