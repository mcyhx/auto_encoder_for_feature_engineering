import pandas as pd
import jieba
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from autoencoder_torch_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train function
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    
    print("Training Begin...")
    for epoch in range(num_epochs):
        
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len,_ = [x.to(device) for x in batch]
            Y, Y_valid_len = X, X_valid_len
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      
            
            num_tokens = Y_valid_len.sum()
            optimizer.step()

#Data prepration

train_data = pd.read_csv('/content/textClassification/data/train_clean.csv')
dev_data = pd.read_csv('/content/textClassification/data/dev_clean.csv')
test_data = pd.read_csv('/content/textClassification/data/train_clean.csv')

data = pd.concat([train_data,dev_data,test_data])
data["text"] = data["text"].apply(query_cut)
data['text'] = data["text"].apply(lambda x: " ".join(x))

vocab = Vocab.build(data)
data['text'] = data["text"].apply(lambda x: vocab.convert_tokens_to_ids(x))

dataset = BookDataset(data)
data_loader = DataLoader(dataset, batch_size=32,collate_fn = collate_fn,shuffle=True)



# hyperparameter config
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
maxlen = 200

#model building
 
encoder = Seq2SeqEncoder(len(vocab), embed_size, num_hiddens, num_layers,maxlen,
                        dropout)
decoder = Seq2SeqDecoder(len(vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = EncoderDecoder(encoder, decoder)


if __name__ == '__main__':
    train_seq2seq(net, data_loader, lr = 0.001, num_epochs = 3, tgt_vocab =vocab, device = device)
    torch.save(encoder.state_dict(), '/Bookclassification/encoder_state_dict')
    torch.save(encoder, '/Bookclassification/encoder')
    torch.save(net, '/Bookclassification/net')
    torch.save(vocab,'/Bookclassification/vocab')
     










