import wikipediaapi
import json
import time
from tqdm import tqdm

# Initialize the Wikipedia API
user_agent = "testing-statistics-app/1.0 (arjunkirtipatel@gmail.com)"
wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")


# Function to get the content of a Wikipedia page
def get_wikipedia_page(title):
    page = wiki_wiki.page(title)
    if page.exists():
        return page
    else:
        return None


# Function to parse the Wikipedia page and extract relevant information
def parse_wikipedia_page(page):
    sections = {}
    for section in page.sections:
        sections[section.title] = section.text

    return {"title": page.title, "summary": page.summary, "sections": sections}


# Function to get all linked pages from a given Wikipedia page
def get_linked_pages(title):
    page = get_wikipedia_page(title)
    if not page:
        return []

    linked_pages = list(page.links.keys())
    return linked_pages


# Main page with links to statistical tests and concepts
# main_page_title = "List of statistics articles"
# a good set of articles to use: https://en.wikipedia.org/wiki/List_of_statistics_articles

main_page_title = "Outline of statistics"
# Get all linked pages from the main page
linked_pages = get_linked_pages(main_page_title)

# Estimate the number of links to be scraped
num_links = len(linked_pages)
print(f"Estimated number of links to be scraped: {num_links}")

# Respect the 150 requests per second limit
delay_between_requests = 1 / 150

# Process each linked page with tqdm progress bar
for linked_page in tqdm(linked_pages, desc="Scraping Wikipedia pages"):
    page = get_wikipedia_page(linked_page)
    if page:
        parsed_page = parse_wikipedia_page(page)
        # Process the parsed page (e.g., save to file, print, etc.)
        with open("wikipedia_articles.jsonl", "a") as file:
            file.write(json.dumps(parsed_page) + "\n")
    time.sleep(delay_between_requests)  # Delay to respect the 150 rps limit
