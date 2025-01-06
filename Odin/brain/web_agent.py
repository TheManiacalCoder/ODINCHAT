import os
import logging
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
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

    def visit_url_with_selenium(self, url):
        """
        Visit the URL using Selenium and Chrome, and return the page source.
        """
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)

        try:
            # Initialize the Chrome driver
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)  # Set timeout for page load
            logging.info(f"Visiting URL: {url}")
            driver.get(url)  # Open the URL in Chrome
            time.sleep(5)  # Wait for the page to load (adjust as needed)

            # Get the page source after it has loaded
            page_source = driver.page_source
            driver.quit()  # Close the browser
            return page_source
        except (TimeoutException, WebDriverException) as e:
            logging.error(f"Error visiting URL {url}: {str(e)}")
            return None

    def scrape_website_content(self, url):
        """
        Scrape the content of a website and extract relevant information.
        """
        # Visit the URL and get the page source
        page_source = self.visit_url_with_selenium(url)
        if not page_source:
            return None

        try:
            # Parse the page source with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')

            # Extract the title
            title = soup.title.string if soup.title else "No Title"

            # Extract all text content
            text_content = soup.get_text(separator="\n")

            # Extract specific elements (e.g., headings, paragraphs)
            headings = {
                'h1': [h1.text for h1 in soup.find_all('h1')],
                'h2': [h2.text for h2 in soup.find_all('h2')],
                'h3': [h3.text for h3 in soup.find_all('h3')],
            }
            paragraphs = [p.text for p in soup.find_all('p')]

            # Organize the scraped content
            scraped_data = {
                'title': title,
                'text_content': text_content,
                'headings': headings,
                'paragraphs': paragraphs,
            }
            return scraped_data
        except Exception as e:
            logging.error(f"Error parsing website content: {str(e)}")
            return None

    def save_scraped_content(self, url, content):
        """
        Save the scraped content to the database.
        """
        if not content:
            return "No content to save."

        try:
            # Convert the content to a string (e.g., JSON format)
            content_str = json.dumps(content, ensure_ascii=False)

            # Generate an embedding for the content
            embedding = self.memory_handler.sentence_to_vec(content_str)

            # Save to the database
            conn = sqlite3.connect(self.conversation_manager.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO embeddings (timestamp, content, embedding, source_type)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                content_str,
                json.dumps(embedding.tolist()) if embedding is not None else None,
                "website"  # Indicate that this is website content
            ))
            conn.commit()
            conn.close()
            logging.info(f"Scraped content saved to the database for URL: {url}")
            return "Scraped content saved successfully."
        except Exception as e:
            logging.error(f"Error saving scraped content: {str(e)}")
            return "An error occurred while saving the scraped content."

    def process_webpage(self, url):
        """
        Process a webpage by visiting it with Selenium, scraping its content, and saving it to the database.
        """
        try:
            # Scrape the webpage content
            scraped_content = self.scrape_website_content(url)
            if not scraped_content:
                return "Failed to scrape the webpage."

            # Save the scraped content to the database
            result = self.save_scraped_content(url, scraped_content)
            return result
        except Exception as e:
            logging.error(f"Error processing webpage: {str(e)}")
            return "An error occurred while processing the webpage."