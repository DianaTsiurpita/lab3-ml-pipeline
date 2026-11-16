"""Generate a readable CI summary from collected JSON metrics."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='metrics/')
    parser.add_argument('--model-metrics', default='artifacts/metrics.json')
    parser.add_argument('--out', default='metrics/report.md')
    args = parser.parse_args()
    folder = Path(args.metrics)
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}
    model = read(Path(args.model_metrics))
    latency = read(folder / 'latency.json')
    verify = read(folder / 'verify.json')
    lines = ['# ML Pipeline — звіт', '',
             f"- Commit: `{os.getenv('GITHUB_SHA', 'local')}`",
             f"- Event: `{os.getenv('GITHUB_EVENT_NAME', 'local')}`",
             f"- Epochs: {model.get('epochs', 'N/A')}",
             f"- Accuracy: {model.get('accuracy', 'N/A')}%",
             f"- Model size: {model.get('model_size_kb', 'N/A')} KB",
             f"- Golden samples: {verify.get('passed', 'N/A')}/{verify.get('total', 'N/A')}",
             '', '## Latency (ms)', '', '| Metric | Mean | p50 | p95 |',
             '|---|---:|---:|---:|']
    for name, key in [('Client', 'client_ms'), ('Server', 'server_ms')]:
        data = latency.get(key, {})
        lines.append(f"| {name} | {data.get('mean', 'N/A')} | "
                     f"{data.get('p50', 'N/A')} | {data.get('p95', 'N/A')} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n')
    print(out.read_text())


if __name__ == '__main__':
    main()
