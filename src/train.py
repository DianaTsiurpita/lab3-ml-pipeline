import argparse
import json
import random
import time
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset
from src.model import SpeechCNN, CLASSES, preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='artifacts')
    parser.add_argument('--classes', nargs='+', default=CLASSES)
    args = parser.parse_args()
    if args.epochs < 1 or args.classes != CLASSES:
        parser.error('Use epochs >= 1 and --classes yes no up down')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    def dataset(split):
        full = torchaudio.datasets.SPEECHCOMMANDS(root='/data', url='speech_commands_v0.02',
                                                   download=True, subset=split)
        ids = [i for i, path in enumerate(full._walker)
               if Path(path).parent.name in CLASSES]
        return Subset(full, ids)
    def collate(batch):
        return torch.stack([preprocess(w, sr) for w, sr, label, _, _ in batch]), torch.tensor(
            [CLASSES.index(label) for _, _, label, _, _ in batch])
    train = DataLoader(dataset('training'), batch_size=32, shuffle=True, collate_fn=collate)
    test = DataLoader(dataset('testing'), batch_size=32, collate_fn=collate)
    model = SpeechCNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    started = time.monotonic()
    history = []
    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for x, y in train:
            opt.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            correct += (output.argmax(1) == y).sum().item()
            total += len(y)
        row = {'epoch': epoch + 1, 'loss': round(loss_sum / len(train), 5),
               'train_accuracy': round(correct / total, 5)}
        history.append(row)
        print(row, flush=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    torch.save(model.state_dict(), out / 'model.pt')
    (out / 'labels.json').write_text(json.dumps(CLASSES))
    metrics = {'accuracy': round(100 * correct / total, 3), 'test_samples': total,
               'epochs': args.epochs, 'seed': args.seed,
               'training_seconds': round(time.monotonic() - started, 2),
               'model_size_bytes': (out / 'model.pt').stat().st_size,
               'model_size_kb': round((out / 'model.pt').stat().st_size / 1024, 2),
               'model_version': '3.0.0', 'architecture': 'SpeechCNN'}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    print('METRICS', metrics, flush=True)

if __name__ == '__main__':
    main()
