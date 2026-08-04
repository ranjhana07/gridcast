from __future__ import annotations

import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import requests

UCI_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
UCI_MEMBER = "LD2011_2014.txt"

Opener = Callable[[str], Iterator[bytes]]


def stream_http(url: str) -> Iterator[bytes]:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        yield from response.iter_content(chunk_size=1 << 20)


def safe_extract(archive: Path, member: str, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as bundle:
        if member not in bundle.namelist():
            raise FileNotFoundError(f"{member} is not in {archive.name}")
        target = destination / Path(member).name
        with bundle.open(member) as source, open(target, "wb") as sink:
            while chunk := source.read(1 << 20):
                sink.write(chunk)
    return target


def fetch_uci(raw_dir: Path, url: str = UCI_URL, opener: Opener = stream_http, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / UCI_MEMBER
    if target.exists() and not force:
        return target
    archive = raw_dir / "electricity_load_diagrams.zip"
    with open(archive, "wb") as sink:
        for chunk in opener(url):
            sink.write(chunk)
    extracted = safe_extract(archive, UCI_MEMBER, raw_dir)
    archive.unlink()
    return extracted
