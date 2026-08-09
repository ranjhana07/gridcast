import io
import zipfile

import pytest

from gridcast.data.download import UCI_MEMBER, fetch_uci


def fake_archive(member: str = UCI_MEMBER) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(member, '"";"MT_001"\n"2012-01-01 00:15:00";1,0\n')
        bundle.writestr("__MACOSX/._" + member, "junk")
    return buffer.getvalue()


def test_fetch_extracts_the_load_file_and_removes_the_archive(tmp_path):
    payload = fake_archive()
    path = fetch_uci(tmp_path, opener=lambda url: iter([payload[:50], payload[50:]]))
    assert path.name == UCI_MEMBER
    assert path.read_text().startswith('"";"MT_001"')
    assert [item.name for item in tmp_path.iterdir()] == [UCI_MEMBER]


def test_existing_file_is_not_downloaded_again(tmp_path):
    (tmp_path / UCI_MEMBER).write_text("cached")

    def explode(url):
        raise AssertionError("should not download")

    assert fetch_uci(tmp_path, opener=explode).read_text() == "cached"


def test_missing_member_is_reported(tmp_path):
    payload = fake_archive("other.txt")
    with pytest.raises(FileNotFoundError):
        fetch_uci(tmp_path, opener=lambda url: iter([payload]))
