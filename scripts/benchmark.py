"""Measure client HTTP latency and reported model inference latency."""
import argparse
import json
import statistics
import time
from pathlib import Path
from urllib import request


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--sample', default='samples/yes.wav')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--out', default='metrics/latency.json')
    args = parser.parse_args()
    if args.n < 1 or args.warmup < 0:
        parser.error('n must be >= 1 and warmup >= 0')
    boundary = 'benchmarkboundary'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="sample.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    body += Path(args.sample).read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    client, server = [], []
    for i in range(args.warmup + args.n):
        req = request.Request(args.url + '/predict', data=body,
                              headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        start = time.perf_counter()
        with request.urlopen(req, timeout=30) as response:
            result = json.load(response)
        elapsed = (time.perf_counter() - start) * 1000
        if i >= args.warmup:
            client.append(elapsed)
            server.append(result['latency_ms'])
    def stats(xs):
        return {k: round(v, 3) for k, v in {
            'mean': statistics.mean(xs), 'p50': percentile(xs, .5),
            'p95': percentile(xs, .95)}.items()}
    output = {'requests': args.n, 'warmup': args.warmup,
              'client_ms': stats(client), 'server_ms': stats(server)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(out.read_text())


if __name__ == '__main__':
    main()
