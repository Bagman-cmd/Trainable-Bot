import json
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os



DATA_FILE = 'training_data.json'



def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        return [
            ("Hello", "Hello! How can I help you?"),
            ("How are you?", "I'm doing great! And you?"),
            ("What are you doing?", "I'm talking to you!"),
            ("What's your name?", "My name is Zero!"),
            ("Bye", "See you! Have a nice day!")
        ]
    


def save_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



def split_data(data):
    questions = [item[0] for item in data]
    answers = [item[1] for item in data]
    return questions, answers



nltk.download('punkt')



def train_bot(data):
    questions, answers = split_data(data)
    model_pipeline = make_pipeline(CountVectorizer(), MultinomialNB())
    model_pipeline.fit(questions, answers)
    return model_pipeline



training_data = load_data()



model = train_bot(training_data)



def chatbot_response(user_input):
    user_input_vector = model.named_steps['countvectorizer'].transform([user_input])
    if user_input_vector.nnz == 0:
        return None
    else:
        predictions = model.predict([user_input])
        return predictions.tolist()



def learn_new_data(user_input, correct_response):
    global training_data
    training_data.append((user_input, correct_response))
    save_data(training_data)
    global model
    model = train_bot(training_data)
    print("Thanks! I remembered that.")



def get_best_response(responses):
    if responses:
        return random.choice(responses)
    else:
        return None



print("Hi! This is a trainable bot. Write 'bye' to exit.")
while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'bye':
        print("Zero: Goodbye!")
        break
    
    responses = chatbot_response(user_input)
    
    if responses:
        best_response = get_best_response(responses)
        print("Zero:", best_response)
    else:
        print("Zero: I do not know how to answer this. Can you help me learn?")
        correct_response = input("You: How was I supposed to respond? ")
        learn_new_data(user_input, correct_response)
