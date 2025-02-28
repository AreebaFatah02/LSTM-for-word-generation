# areeba fatah (21i-0349)
from flask import Flask, request, jsonify, render_template
import torch
import pickle
import torch.nn as nn


# LSTM same class as in code
class Shakespeare_LSTM(nn.Module):
    def __init__(self,vocab_length,embedding_dimensions,hidden_dimensions):
        super(Shakespeare_LSTM, self).__init__()
        self.shakespeare_embedding=nn.Embedding(vocab_length,embedding_dimensions)
        self.shakespeare_lstm=nn.LSTM(embedding_dimensions,hidden_dimensions,batch_first=True)
        self.shakespeare_linear=nn.Linear(hidden_dimensions,vocab_length)
        
    def forward(self,words):
        words=self.shakespeare_embedding(words)
        words,_=self.shakespeare_lstm(words)
        words=self.shakespeare_linear(words[:,-1,:]) 
        return words

    
vocabulary_size =27222  
shakespeare_model=Shakespeare_LSTM(27222,64,128)
shakespeare_model.load_state_dict(torch.load('lstm_shakespeare_model.pt',map_location=torch.device('cpu'),weights_only=True))
shakespeare_model.eval()


# loading vocab
with open('vocabulary.pkl','rb') as shakespeare_vocab_reader:
    shakespeare_vocab=pickle.load(shakespeare_vocab_reader)

    
def words_to_sequence(words):
    return [shakespeare_vocab.get(word,shakespeare_vocab.get('<UNK>',0)) for word in words] 


# for redirecting to index which is main page
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


# this is used for post to webpage based on model's output
@app.route('/predict',methods=['POST'])
def predict():
    input_text=request.json['text']
    words=input_text.split()
    sequence=words_to_sequence(words)[-16:]
    sequence_tensor=torch.tensor(sequence,dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output=shakespeare_model(sequence_tensor)
        predicted_word_index=output.argmax(dim=1).item()

    predicted_word=[k for k, v in shakespeare_vocab.items() if v == predicted_word_index][0]
    return jsonify({'predicted_word':predicted_word})


if __name__ == '__main__':
    app.run(debug=True)
