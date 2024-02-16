import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from src.modules import Transformer


def generate_sample_data(
    src_vocab_size, tgt_vocab_size, max_seq_len, num_samples
):
    src_data = torch.randint(1, src_vocab_size, (num_samples, max_seq_len))
    tgt_data = torch.randint(1, tgt_vocab_size, (num_samples, max_seq_len))

    return src_data, tgt_data


def train(model, src_data, tgt_data, tgt_vocab_size):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=1e-09
    )

    model.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()

        print("Epoch: {}, Loss: {}".format(epoch + 1, loss.item()))


def main(args):
    src_data, tgt_data = generate_sample_data(
        args.src_vocab_size,
        args.tgt_vocab_size,
        args.max_seq_len,
        args.num_samples,
    )

    transformer = Transformer(
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        d_model=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    train(transformer, src_data, tgt_data, args.tgt_vocab_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train an encoder-decoder Transformer.")

    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--src_vocab_size", type=int, default=500)
    parser.add_argument("--tgt_vocab_size", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=64)

    print(parser.parse_args())

    main(parser.parse_args())
