# This script will upsert the chunked wikipedia articles into Pinecone.
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from tqdm import tqdm
import time 
from agent_configs import STATWIKI_INDEX, EMBEDDING_MODEL, TOP_K, PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = STATWIKI_INDEX


def count_tokens(text):
    return len(text) / 4



# Function to embed chunks in groups of 96
def embed_chunks_in_batches(data, pc):
    embeddings = []
    batch_size = 96
    total_tokens = 0

    # Process data in batches of 96, respecting rate limits
    for i in tqdm(range(0, len(data), batch_size), desc="Embedding chunks"):
        batch = data[i:i + batch_size]  # Get the current batch
        batch_tokens = sum(count_tokens(d) for d in batch)  # Calculate total tokens in batch
        total_tokens += batch_tokens  # Update total tokens processed
        # Check if we've exceeded the token limit
        if total_tokens > 250000:
            time.sleep(60)  # Wait for a minute to reset the token limit
            total_tokens = batch_tokens  # Reset total tokens
        batch_embeddings = pc.inference.embed(
            "multilingual-e5-large",
            inputs=batch,  # Use 'Chunk Content' for embedding
            parameters={
                "input_type": "passage", "truncate": "END"
            }
        )
        embeddings.extend(batch_embeddings)  # Collect the embeddings

        # Check if we've exceeded the request limit
        if len(embeddings) % 200 == 0:
            time.sleep(60)  # Wait for a minute to reset the request limit

    return embeddings






if __name__ == "__main__":
    # Read the data
    df = pd.read_csv('article_chunks.csv')

    # create index

    if index_name not in pc.list_indexes().names():    
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    index = pc.Index(index_name)

    # Embed the chunk content
    embeddings = embed_chunks_in_batches(df['Chunk Content'].tolist(), pc)
    embeddings = [e["values"] for e in embeddings]

    # Create a new dataframe with the required format
    df['values'] = embeddings
    df['id'] = df.index.astype(str)
    df.fillna('', inplace=True)

    print(df.head())
    df['metadata'] = df[['Article Title', 'Section Title', 'Chunk Content', 'Chunk Number']].apply(lambda row: row.to_dict(), axis=1)

    # Select the required columns
    df = df[['id', 'values', 'metadata']]
    print(df.metadata[0])
    # write to file 
    df.to_csv('embedded_articles.csv', index=False)
 
    # Upsert the data into Pinecone
    index.upsert_from_dataframe(df, batch_size=50)

