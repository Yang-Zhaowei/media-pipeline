"""Opt-in, CPU-only acceptance of an already-built wheel, without networking.

Set REGRAIN_TEST_WHEEL to its absolute path. The tests install it into a clean
temporary Controller environment and never import production code from checkout.
The model stages below are lightweight subprocess fakes, not GPU validation.
"""

from __future__ import annotations

import configparser
import json
import os
import subprocess
import venv
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path

import pytest


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(command: list[str | Path], cwd: Path) -> str:
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=_clean_env(),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@pytest.fixture(scope="module")
def wheel() -> Path:
    configured = os.environ.get("REGRAIN_TEST_WHEEL")
    if not configured:
        pytest.skip("set REGRAIN_TEST_WHEEL to an existing wheel for installed acceptance")
    path = Path(configured).resolve()
    assert path.is_file() and path.suffix == ".whl", f"not a wheel file: {path}"
    return path


@dataclass(frozen=True)
class Controller:
    python: Path
    package: Path
    cwd: Path

    def cli(self, name: str) -> Path:
        return self.python.parent / (name + ".exe" if os.name == "nt" else name)


@pytest.fixture(scope="module")
def controller(wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Controller:
    root = tmp_path_factory.mktemp("regrain-installed")
    environment = root / "controller"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = _python(environment)
    _run(
        [python, "-m", "pip", "--disable-pip-version-check", "install",
         "--no-index", "--no-deps", wheel],
        root,
    )
    package = Path(_run(
        [python, "-c", "import media_pipeline; print(media_pipeline.__file__)"], root,
    ).strip()).parent
    assert package.is_relative_to(environment)
    return Controller(python=python, package=package, cwd=root)


def test_wheel_metadata_and_only_production_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_path, = [name for name in names if name.endswith(".dist-info/METADATA")]
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        assert metadata["Name"] == "regrain"
        assert metadata["Version"] == "0.1.0"
        assert metadata["Requires-Python"] == ">=3.10"
        # Development extras are allowed; the Controller has no required dependencies.
        assert all("extra ==" in requirement.partition(";")[2]
                   for requirement in metadata.get_all("Requires-Dist", []))
        entrypoints_path, = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        entrypoints = configparser.ConfigParser()
        entrypoints.read_string(archive.read(entrypoints_path).decode("utf-8"))
        assert dict(entrypoints["console_scripts"]) == {
            "regrain": "media_pipeline.cli:main",
            "media-pipeline": "media_pipeline.cli:main",
        }
        assert not any({"tests", "experiments", "fixtures"}.intersection(Path(name).parts)
                       for name in names)
        for relative in ("SKILL.md", "examples/custom-voice.json", "examples/clone.json"):
            assert "media_pipeline/skills/regrain-speech/" + relative in names


def test_installed_cli_compatibility_and_skill_examples(controller: Controller) -> None:
    for option in ("--help", "--version"):
        primary = _run([controller.cli("regrain"), option], controller.cwd)
        assert primary == _run([controller.cli("media-pipeline"), option], controller.cwd)
        assert "regrain" in primary
        if option == "--version":
            assert primary.strip() == "regrain 0.1.0"
        else:
            assert "speech" in primary
    inspection = json.loads(_run([controller.python, "-c", """
import importlib.metadata
import importlib.util
import json
import media_pipeline
assert media_pipeline.__version__ == importlib.metadata.version('regrain') == '0.1.0'
print(json.dumps({name: importlib.util.find_spec(name) is not None
                  for name in ('torch', 'qwen_tts', 'qwen_asr', 'soundfile')}))
"""], controller.cwd))
    assert not any(inspection.values())
    examples = controller.package / "skills" / "regrain-speech" / "examples"
    for filename, mode in (("custom-voice.json", "custom"), ("clone.json", "clone")):
        example = examples / filename
        payload = json.loads(example.read_text(encoding="utf-8"))
        if mode == "clone":
            assert not (example.parent / payload["voice"]["asset"]).exists()
        result = json.loads(_run(
            [controller.cli("regrain"), "speech", "validate", example], controller.cwd,
        ))
        assert result["status"] == "valid"
        assert result["plan"]["segments"]
        assert ("voice" in result["plan"]) == (mode == "clone")
    legacy_help = _run(
        [controller.python, "-m", "media_pipeline", "--help"], controller.cwd,
    )
    assert "Compile original text and forced alignment into SRT captions" in legacy_help


_WORKER_COMMON = r"""
import json
import sys
from pathlib import Path
import media_pipeline
import _regrain_packaging_probe as probe

spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
expected_package, task = spec['args']
assert Path(media_pipeline.__file__).resolve().parent == Path(expected_package).resolve()
assert media_pipeline.__version__ == '0.1.0'
assert Path(probe.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
segments_dir = Path(task['run_dir']) / 'segments'
segments_dir.mkdir(parents=True, exist_ok=True)
results = []
"""

_TTS_WORKER = _WORKER_COMMON + r"""
assert probe.IDENTITY == 'tts'
import math
from media_pipeline.postprocess import write_wav
samples = [int(7000 * math.sin(index / 80)) for index in range(24000)]
for segment in task['segments']:
    write_wav(segments_dir / (segment['safe_name'] + '.wav'), samples, 24000)
    results.append({'segment_id': segment['id'], 'status': 'ok'})
Path(spec['manifest']).write_text(json.dumps({'runtime_error': None, 'segments': results}),
                                  encoding='utf-8')
"""

_ALIGNMENT_WORKER = _WORKER_COMMON + r"""
assert probe.IDENTITY == 'alignment'
import unicodedata
for segment in task['segments']:
    spoken = [char for char in segment['text']
              if not char.isspace() and unicodedata.category(char)[0] not in 'PZ']
    step = 0.7 / len(spoken)
    records = [{'text': char, 'start': 0.2 + index * step,
                'end': 0.2 + (index + 1) * step} for index, char in enumerate(spoken)]
    path = segments_dir / (segment['safe_name'] + '.alignment.raw.json')
    path.write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    results.append({'segment_id': segment['id'], 'status': 'ok'})
Path(spec['manifest']).write_text(json.dumps({'runtime_error': None, 'segments': results}),
                                  encoding='utf-8')
"""

_CONTROLLER_RUN = r"""
import json
import sys
from pathlib import Path
import media_pipeline
from media_pipeline.render import render_speech, _run_model_stage

config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert Path(media_pipeline.__file__).resolve().parent == Path(config['package']).resolve()
def task(stage):
    return lambda request: _run_model_stage(
        interpreter=config[stage + '_python'], stage_script=config[stage + '_script'],
        args=(config['package'], request),
    )
result = render_speech(
    config['speech'], config['output'], tts_python=config['tts_python'],
    alignment_python=config['alignment_python'], tts_model='cpu-fake',
    alignment_model='cpu-fake', max_segment_chars=200,
    _wav_task=task('tts'), _align_task=task('alignment'),
)
assert result.status == 'complete'
print(result.status)
"""


def test_installed_controller_pins_package_and_preserves_worker_dependencies(
    controller: Controller, tmp_path: Path,
) -> None:
    config = {"package": str(controller.package)}
    # A neighboring Controller dependency must not enter either worker's path.
    (controller.package.parent / "_regrain_packaging_probe.py").write_text(
        "IDENTITY = 'controller'\n", encoding="utf-8",
    )
    for stage in ("tts", "alignment"):
        environment = tmp_path / stage
        venv.EnvBuilder(with_pip=False).create(environment)
        python = _python(environment)
        site_packages = Path(json.loads(_run(
            [python, "-c", "import json, sysconfig; print(json.dumps(sysconfig.get_path('purelib')))"],
            tmp_path,
        )))
        (site_packages / "_regrain_packaging_probe.py").write_text(
            f"IDENTITY = {stage!r}\n", encoding="utf-8",
        )
        stale_package = site_packages / "media_pipeline"
        stale_package.mkdir()
        (stale_package / "__init__.py").write_text(
            "raise AssertionError('stale worker package was imported')\n", encoding="utf-8",
        )
        config[stage + "_python"] = str(python)
    speech = tmp_path / "speech.json"
    speech.write_text(json.dumps({
        "language": "Chinese", "speaker": "Uncle_Fu", "segments": [
            {"id": "opening", "text": "欢迎收听。", "pause_after_ms": 125},
            {"id": "next", "text": "我们开始。"},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "rendered"
    config.update(speech=str(speech), output=str(output),
                  tts_script=_TTS_WORKER, alignment_script=_ALIGNMENT_WORKER)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    driver = tmp_path / "render.py"
    driver.write_text(_CONTROLLER_RUN, encoding="utf-8")
    assert _run([controller.python, driver, config_path], tmp_path).strip() == "complete"
    for artifact in ("final.wav", "final.srt", "timeline.json", ".complete"):
        assert (output / "final" / artifact).is_file()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["completed_stages"] == [
        "preflight", "tts", "alignment", "postprocess", "captions", "assemble",
    ]
    timeline = json.loads((output / "final" / "timeline.json").read_text(encoding="utf-8"))
    assert [segment["segment_id"] for segment in timeline["segments"]] == ["opening", "next"]
    assert timeline["total_frames"] > 0
    assert "欢迎收听" in (output / "final" / "final.srt").read_text(encoding="utf-8")
