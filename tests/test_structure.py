from pathlib import Path
import ast


def test_required_sources():
    for name in ['src/train.py', 'src/model.py', 'src/api.py',
                 'scripts/verify_model.py', 'scripts/benchmark.py', 'scripts/make_report.py']:
        ast.parse(Path(name).read_text())


def test_expected_labels():
    import json
    expected = json.loads(Path('samples/expected.json').read_text())
    assert expected == {f'{name}.wav': name for name in ['yes', 'no', 'up', 'down']}
