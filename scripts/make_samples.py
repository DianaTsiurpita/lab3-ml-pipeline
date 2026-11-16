"""Generate fixed test-set examples; never use training-set audio."""
import json
from pathlib import Path
import torchaudio
from src.model import CLASSES


def main():
    root = Path('/samples')
    root.mkdir(parents=True, exist_ok=True)
    ds = torchaudio.datasets.SPEECHCOMMANDS(
        root='/data', url='speech_commands_v0.02', download=True, subset='testing'
    )
    found = set()
    for path in sorted(ds._walker):
        label = Path(path).parent.name
        if label in CLASSES and label not in found:
            wave, sr = torchaudio.load(path)
            torchaudio.save(str(root / f'{label}.wav'), wave, sr)
            found.add(label)
        if len(found) == len(CLASSES):
            break
    if found != set(CLASSES):
        raise RuntimeError('Missing test samples')
    (root / 'expected.json').write_text(json.dumps({f'{x}.wav': x for x in CLASSES}, indent=2))


if __name__ == '__main__':
    main()
