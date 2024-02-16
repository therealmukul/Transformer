import torch
import torch.nn as nn
import torch.optim as optim

from src.modules import Transformer


def generate_sample_data(src_vocab_size, tgt_vocab_size, max_seq_len,
                         num_samples):
    src_data = torch.randint(1, src_vocab_size, (num_samples, max_seq_len))
    tgt_data = torch.randint(1, tgt_vocab_size, (num_samples, max_seq_len))

    return src_data, tgt_data


def train(model, src_data, tgt_data):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99),
                           eps=1e-09)

    model.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1)
        )
        loss.backward()
        optimizer.step()

        print('Epoch: {}, Loss: {}'.format(epoch+1, loss.item()))


if __name__ == '__main__':
    print('STARTING TRAINING')

    src_vocab_size, tgt_vocab_size = 5000, 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 100
    dropout = 0.1

    src_data, tgt_data = generate_sample_data(
        src_vocab_size,
        tgt_vocab_size,
        max_seq_len,
        num_samples=64
    )

    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    )

    train(transformer, src_data, tgt_data)
