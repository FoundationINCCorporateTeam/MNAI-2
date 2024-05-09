from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bs4 import BeautifulSoup
import requests
import re
import Levenshtein

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# Function to search Wikipedia and select the closest article
def search_and_select_article(query):
    try:
        # Search Wikipedia
        search_url = f"https://en.wikipedia.org/w/index.php?search={query.replace(' ', '+')}"
        response = requests.get(search_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Get all search result titles and URLs
            search_results = soup.find_all("div", class_="mw-search-result-heading")
            titles = [result.a.text for result in search_results]
            urls = ["https://en.wikipedia.org" + result.a.get("href") for result in search_results]
            # Find the title most similar to the query
            best_match_index = min(range(len(titles)), key=lambda i: Levenshtein.distance(query.lower(), titles[i].lower()))
            return urls[best_match_index]
        else:
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None

# Function to fetch article content from Wikipedia
def fetch_wikipedia_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = " ".join([p.get_text() for p in paragraphs])
            return content
        else:
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None

# Function to generate a summary using GPT-2
def generate_summary(text, answer, max_length=150):
    inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    inputs = inputs[:, :max_length]
    summary_ids = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.replace(answer, f"**{answer}**")

# Main loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    else:
        # Search Wikipedia and select the closest article
        article_url = search_and_select_article(user_input.strip())
        if article_url:
            # Fetch article content
            content = fetch_wikipedia_content(article_url)
            if content:
                # Generate summary and extract answer
                answer = re.search(r'\b{}\b'.format(re.escape(user_input)), content, re.IGNORECASE)
                if answer:
                    summary = generate_summary(content, user_input)
                    print("Bot (Summary):", summary)
                else:
                    print("Bot:", "Sorry, I couldn't find an answer to your question.")
            else:
                print("Bot:", "Sorry, I couldn't fetch information about that topic from Wikipedia.")
        else:
            print("Bot:", "Sorry, I couldn't find information about that topic on Wikipedia.")
