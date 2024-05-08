from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wikipedia

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Retrieve information from Wikipedia based on user query
def get_wikipedia_summary(topic):
    try:
        summary = wikipedia.summary(topic)
        return summary
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find information on that topic."
    except wikipedia.exceptions.DisambiguationError as e:
        options = ", ".join(e.options[:5])  # Display first 5 options
        return f"Please be more specific. Did you mean: {options}?"

# Generate response based on user input
def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
user_input = "Tell me about Artificial Intelligence"
topic = "Artificial Intelligence"
wikipedia_summary = get_wikipedia_summary(topic)
bot_response = generate_response(user_input)

print("Wikipedia Summary:", wikipedia_summary)
print("Chatbot Response:", bot_response)
