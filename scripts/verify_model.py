"""Check API predictions against independent, fixed test samples."""
import argparse
import json
from pathlib import Path
from urllib import request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--expected', default='samples/expected.json')
    parser.add_argument('--out', default='metrics/verify.json')
    args = parser.parse_args()
    expected = json.loads(Path(args.expected).read_text())
    results = []
    for filename, label in expected.items():
        audio = (Path(args.expected).parent / filename).read_bytes()
        boundary = 'speechcommandsboundary'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n').encode()
        body += audio + f'\r\n--{boundary}--\r\n'.encode()
        req = request.Request(args.url + '/predict', data=body, method='POST',
                              headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        with request.urlopen(req, timeout=30) as response:
            result = json.load(response)
        predicted = result['label']
        results.append({'file': filename, 'expected': label, 'predicted': predicted,
                        'passed': predicted == label})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'passed': sum(x['passed'] for x in results),
                               'total': len(results), 'results': results}, indent=2))
    print(out.read_text())
    if not all(x['passed'] for x in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
