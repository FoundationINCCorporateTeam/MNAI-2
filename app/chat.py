from flask import Flask, send_file, request, jsonify
from bs4 import BeautifulSoup
import requests
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

app = Flask(__name__)

# Enable CORS for all routes
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/')
def index():
    return send_file('static/botty.html')

@app.route('/get-response', methods=['POST'])
def get_response():
    user_input = request.form.get('question', '')
    if not user_input:
        return jsonify({"error": "No user input provided."}), 400
    
    # Extract important phrases from user input
    phrases = extract_phrases(user_input)
    
    # Search Wikipedia for relevant articles based on extracted phrases
    relevant_articles = search_wikipedia(phrases)
    
    # Extract information from the first relevant article
    if relevant_articles:
        article_url = relevant_articles[0]  # Using the first relevant article for simplicity
        summary = extract_information(article_url)
        response_data = {"article_url": article_url, "summary": summary}
    else:
        response_data = {"error": "I'm sorry, I couldn't find relevant information for your query."}
    
    return add_cors_headers(jsonify(response_data))

# Function to extract important phrases from user input using GPT-2
def extract_phrases(user_input):
    # Tokenize the input text
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    
    # Generate output using GPT-2
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove leading/trailing whitespaces and split into phrases
    phrases = [phrase.strip() for phrase in output_text.split(".") if phrase.strip()]
    
    return phrases

# Function to search Wikipedia for relevant articles based on extracted phrases
def search_wikipedia(phrases):
    # Perform a search for each phrase on Wikipedia
    articles = []
    for phrase in phrases:
        search_url = f"https://en.wikipedia.org/w/index.php?search={phrase}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the first search result (assuming it's the most relevant)
        search_results = soup.find_all('div', class_='mw-search-result-heading')
        if search_results:
            first_result = search_results[0].a['href']
            article_url = f"https://en.wikipedia.org{first_result}"
            articles.append(article_url)
    
    return articles

# Function to extract relevant information from Wikipedia article
def extract_information(article_url):
    response = requests.get(article_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract the content of the Wikipedia article
    paragraphs = soup.find_all('p')
    summary = ""
    for p in paragraphs[:3]:
        summary += p.text.strip() + "\n"
    
    return summary

if __name__ == '__main__':
    app.run(debug=True)
