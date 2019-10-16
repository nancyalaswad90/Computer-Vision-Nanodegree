import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
       
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    


    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,  dropout=0.2, num_layers=1):
        #pass
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # define embedding layer        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        ## Define the LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        # set class variables

        
        # Define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # second try with softmax added 
        self.softmax = nn.Softmax(dim=2)
        
    
    def forward(self, features, captions):
        #pass
        
#        batch_size = nn_input.size(0)
        batch_size = features.size(0)

#        embeds = self.embedding( features, captions )#nn_input)
        
        #captions = self.embedding(captions)
        captions = self.embedding(captions[:, :-1])

        features = features.unsqueeze(1)
        
        # defining the inputs as the concatenation of the features and captions arguments
        inputs = torch.cat((features, captions), 1)
        lstm_out, _ = self.lstm(inputs, None)
    
        # stack up lstm outputs
#        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        
        # dropout and fully-connected layer
        output = self.fc(lstm_out)
        
        # reshape into (batch_size, seq_length, output_size)

#        return out, hidden
        return output
    def sample(self, inputs, states=None, max_len=20):
        
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # pass
        sentence=[]
#        lstm_state = None
        for i in range(max_len):
            # the state of the lstm is changing so keep track 
            lstm_out, states = self.lstm(inputs, states)
#            lstm_out, lstm_state = self.lstm(inputs, lstm_state)

            # convert LSTM output to word predictions
            lstm_out = lstm_out.squeeze(1)
            outputs = self.fc(lstm_out)
            #p = F.softmax(output, dim=1)#.data
            
            # Returns the indices of the maximum value of all elements in the "output" tensor
            p =  outputs.max(1)[1]

            p_idx = p.item()
            sentence.append(p.item()) 
            
            # the last index has to be 1 to signify the 'end'
            if p_idx == 1:
                break

            inputs = self.embedding(p).unsqueeze(1)
        
        return sentence