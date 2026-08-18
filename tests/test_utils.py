import json
import os

from backend.utils import (
    bytes_to_gb,
    bytes_to_mb,
    extract_directive,
    failure,
    file_size_mb,
    guarded,
    is_existing_file,
    read_text_file,
    timestamped_id,
    write_json_file,
    write_text_file,
)


def test_byte_conversions():
    assert bytes_to_gb(1024 ** 3) == 1.0
    assert bytes_to_mb(1024 ** 2 * 3) == 3.0


def test_timestamped_id_prefix():
    ident = timestamped_id("msg")
    assert ident.startswith("msg_")
    assert ident.split("_")[1].isdigit()


def test_text_and_json_roundtrip(tmp_path):
    text_path = os.path.join(str(tmp_path), "nested", "note.txt")
    written = write_text_file(text_path, "hello")
    assert written == 5
    assert read_text_file(text_path) == "hello"
    assert is_existing_file(text_path)
    assert file_size_mb(text_path) == 0.0

    json_path = os.path.join(str(tmp_path), "data.json")
    write_json_file(json_path, {"a": 1})
    assert json.loads(read_text_file(json_path)) == {"a": 1}


def test_missing_file_size_is_zero(tmp_path):
    assert file_size_mb(os.path.join(str(tmp_path), "absent.bin")) == 0.0


def test_guarded_converts_exceptions():
    @guarded
    def boom():
        raise ValueError("nope")

    @guarded
    def ok():
        return {"success": True}

    assert boom() == failure("nope")
    assert ok() == {"success": True}


def test_extract_directive():
    assert extract_directive("prefix [JOURNAL: did work ] suffix", "JOURNAL") == "did work"
    assert extract_directive("no directive here", "JOURNAL") is None
    assert extract_directive("[JOURNAL: unterminated", "JOURNAL") is None
