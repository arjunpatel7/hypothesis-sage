import json
import pandas as pd
from tqdm import tqdm


# Function to count tokens (approximation)
def count_tokens(text):
    return len(text.split())


# Function to chunk text into sections
def chunk_text(text, max_tokens=1024):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if count_tokens(" ".join(current_chunk)) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []  # Reset for the next chunk

    # Add any remaining words as a final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Function to chunk an article
def chunk_article(article, max_tokens=1024):
    chunks = []

    # Chunk the summary
    if "summary" in article:
        chunks.extend(chunk_text(article["summary"], max_tokens))

    # Chunk the sections
    if "sections" in article:
        for _, section_text in article["sections"].items():
            chunks.extend(chunk_text(section_text, max_tokens))

    return chunks


# write a function that checks the file for how many chunks can be in it
# print this number
def check_file(file):
    count = 0
    total_tokens = 0
    for line in tqdm(file):
        article = json.loads(line)
        article_chunks = chunk_article(article)
        count += len(article_chunks)
        for chunk in article_chunks:
            total_tokens += count_tokens(chunk)
    return count, total_tokens


# Example usage
if __name__ == "__main__":

    # Load articles from a JSONL file
    with open("wikipedia_articles.jsonl", "r") as file:
        # print diagnostics of token and chunk count with descriptors
        print("Checking file...")
        chunk_count, total_tokens = check_file(file)  # Store results
        print("Number of chunks: ", chunk_count)
        print("Total tokens: ", total_tokens)

        # Reset file pointer to the beginning for further processing
        file.seek(0)  # Reset file pointer

        # Initialize an empty list to store the data
        data = []

        for line in tqdm(file):
            article = json.loads(line)
            # Chunk the article
            article_chunks = chunk_article(article)

            # Iterate over each chunk and its index within the article
            for chunk_index, chunk in enumerate(article_chunks, start=1):
                # Extract section title if available
                section_title = None
                for section, content in article.get("sections", {}).items():
                    if chunk in content:
                        section_title = section
                        break

                # Append the data to the list
                data.append(
                    {
                        "Article Title": article["title"],
                        "Section Title": section_title,
                        "Chunk Content": chunk,
                        "Chunk Number": chunk_index,
                    }
                )

        # Convert the list to a pandas dataframe
        df = pd.DataFrame(data)

        df.to_csv("article_chunks.csv", index=False)
