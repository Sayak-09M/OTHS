from __future__ import annotations

import math
import re
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass

_MI_INT8 = 1
_MI_UINT8 = 2
_MI_INT16 = 3
_MI_UINT16 = 4
_MI_INT32 = 5
_MI_UINT32 = 6
_MI_SINGLE = 7
_MI_DOUBLE = 9
_MI_INT64 = 12
_MI_UINT64 = 13
_MI_MATRIX = 14
_MI_COMPRESSED = 15
_MI_UTF8 = 16
_MI_UTF16 = 17
_MI_UTF32 = 18

_KNOWN_DATA_TYPES = {
    _MI_INT8,
    _MI_UINT8,
    _MI_INT16,
    _MI_UINT16,
    _MI_INT32,
    _MI_UINT32,
    _MI_SINGLE,
    _MI_DOUBLE,
    _MI_INT64,
    _MI_UINT64,
    _MI_MATRIX,
    _MI_COMPRESSED,
    _MI_UTF8,
    _MI_UTF16,
    _MI_UTF32,
}

_MX_CELL = 1
_MX_STRUCT = 2
_MX_OBJECT = 3
_MX_CHAR = 4
_MX_SPARSE = 5
_MX_FUNCTION = 16
_MX_OPAQUE = 17

_NUMERIC_FORMATS = {
    _MI_INT8: "b",
    _MI_UINT8: "B",
    _MI_INT16: "h",
    _MI_UINT16: "H",
    _MI_INT32: "i",
    _MI_UINT32: "I",
    _MI_SINGLE: "f",
    _MI_DOUBLE: "d",
    _MI_INT64: "q",
    _MI_UINT64: "Q",
}

_IDENTITY_LABEL_TOKENS = {
    "aadhaar",
    "aadhar",
    "abha",
    "epic",
    "identity",
    "licence",
    "license",
    "nic",
    "nid",
    "pan",
    "passport",
    "uidai",
    "voter",
}


class MatlabExtractionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class _Budget:
    max_nodes: int
    max_text_chars: int
    max_decompressed_bytes: int
    max_numeric_values: int
    nodes: int = 0
    text_chars: int = 0
    decompressed_bytes: int = 0

    def consume_node(self) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise MatlabExtractionError("matlab_resource_limit_exceeded")

    def add_text(self, parts: list[str], value: str) -> None:
        cleaned = "".join(
            character
            for character in value.replace("\x00", "")
            if character.isprintable() or character in "\n\t"
        ).strip()
        if not cleaned:
            return
        self.text_chars += len(cleaned)
        if self.text_chars > self.max_text_chars:
            raise MatlabExtractionError("matlab_resource_limit_exceeded")
        parts.append(cleaned)


def _elements(
    payload: bytes | memoryview, endian: str, budget: _Budget
) -> Iterator[tuple[int, memoryview]]:
    view = memoryview(payload)
    cursor = 0
    while cursor < len(view):
        remaining = len(view) - cursor
        if remaining < 8:
            if bytes(view[cursor:]).strip(b"\x00"):
                raise MatlabExtractionError("corrupt_or_unsupported_matlab")
            return
        budget.consume_node()
        first, second = struct.unpack_from(f"{endian}HH", view, cursor)
        if first in _KNOWN_DATA_TYPES and 0 < second <= 4:
            data_type = first
            size = second
            start = cursor + 4
            stop = start + size
            cursor += 8
        else:
            data_type, size = struct.unpack_from(f"{endian}II", view, cursor)
            if data_type not in _KNOWN_DATA_TYPES:
                raise MatlabExtractionError("corrupt_or_unsupported_matlab")
            start = cursor + 8
            stop = start + size
            if stop > len(view):
                raise MatlabExtractionError("corrupt_or_unsupported_matlab")
            cursor = stop if data_type == _MI_COMPRESSED else stop + ((-size) % 8)
            if cursor > len(view) and stop != len(view):
                raise MatlabExtractionError("corrupt_or_unsupported_matlab")
            cursor = min(cursor, len(view))
        yield data_type, view[start:stop]


def _unpack_values(
    data_type: int, payload: memoryview, endian: str
) -> tuple[int | float, ...]:
    format_code = _NUMERIC_FORMATS.get(data_type)
    if format_code is None:
        return ()
    item_size = struct.calcsize(format_code)
    if len(payload) % item_size:
        raise MatlabExtractionError("corrupt_or_unsupported_matlab")
    return tuple(
        value[0]
        for value in struct.iter_unpack(f"{endian}{format_code}", payload)
    )


def _decode_name(data_type: int, payload: memoryview, endian: str) -> str:
    raw = bytes(payload)
    try:
        if data_type in {_MI_INT8, _MI_UINT8, _MI_UTF8}:
            return raw.decode("utf-8", errors="replace")
        if data_type in {_MI_INT16, _MI_UINT16, _MI_UTF16}:
            codec = "utf-16le" if endian == "<" else "utf-16be"
            return raw.decode(codec, errors="replace")
        if data_type == _MI_UTF32:
            codec = "utf-32le" if endian == "<" else "utf-32be"
            return raw.decode(codec, errors="replace")
    except (LookupError, UnicodeError):
        return ""
    return ""


def _decode_char_array(
    data_type: int,
    payload: memoryview,
    dimensions: tuple[int, ...],
    endian: str,
) -> str:
    if data_type in {_MI_UTF8, _MI_UTF16, _MI_UTF32}:
        return _decode_name(data_type, payload, endian)
    values = _unpack_values(data_type, payload, endian)
    if not values:
        return ""
    characters = [
        chr(int(value))
        for value in values
        if isinstance(value, int) and 0 <= value <= 0x10FFFF
    ]
    rows = dimensions[0] if dimensions else 1
    if rows <= 1 or len(characters) % rows:
        return "".join(characters)
    columns = len(characters) // rows
    return "\n".join(
        "".join(characters[row + rows * column] for column in range(columns))
        for row in range(rows)
    )


def _decompress(payload: memoryview, budget: _Budget) -> bytes:
    remaining = budget.max_decompressed_bytes - budget.decompressed_bytes
    if remaining <= 0:
        raise MatlabExtractionError("matlab_resource_limit_exceeded")
    try:
        decompressor = zlib.decompressobj()
        output = decompressor.decompress(payload, remaining + 1)
        if len(output) > remaining or decompressor.unconsumed_tail:
            raise MatlabExtractionError("matlab_resource_limit_exceeded")
        tail = decompressor.flush(remaining - len(output) + 1)
    except zlib.error as error:
        raise MatlabExtractionError("corrupt_or_unsupported_matlab") from error
    if len(output) + len(tail) > remaining:
        raise MatlabExtractionError("matlab_resource_limit_exceeded")
    if not decompressor.eof or decompressor.unused_data:
        raise MatlabExtractionError("corrupt_or_unsupported_matlab")
    decoded = output + tail
    budget.decompressed_bytes += len(decoded)
    return decoded


def _dimensions(
    data_type: int, payload: memoryview, endian: str
) -> tuple[int, ...]:
    values = _unpack_values(data_type, payload, endian)
    if not values or any(not isinstance(value, int) or value < 0 for value in values):
        raise MatlabExtractionError("corrupt_or_unsupported_matlab")
    return tuple(int(value) for value in values)


def _field_names(
    length_element: tuple[int, memoryview],
    names_element: tuple[int, memoryview],
    endian: str,
) -> tuple[str, ...]:
    lengths = _unpack_values(length_element[0], length_element[1], endian)
    if not lengths:
        return ()
    field_length = int(lengths[0])
    if field_length <= 0 or field_length > 4_096:
        raise MatlabExtractionError("corrupt_or_unsupported_matlab")
    raw_names = bytes(names_element[1])
    return tuple(
        raw_names[offset : offset + field_length]
        .split(b"\x00", 1)[0]
        .decode("utf-8", errors="replace")
        for offset in range(0, len(raw_names), field_length)
    )


def _identity_related_label(label: str) -> bool:
    tokens = set(re.sub(r"[^a-z0-9]+", " ", label.casefold()).split())
    return bool(tokens & _IDENTITY_LABEL_TOKENS)


def _extract_matrix(
    payload: memoryview,
    endian: str,
    budget: _Budget,
    parts: list[str],
    *,
    depth: int,
    inherited_label: str = "",
) -> None:
    if depth > 32:
        raise MatlabExtractionError("matlab_resource_limit_exceeded")
    elements = list(_elements(payload, endian, budget))
    if len(elements) < 3:
        raise MatlabExtractionError("corrupt_or_unsupported_matlab")
    flags = _unpack_values(elements[0][0], elements[0][1], endian)
    if not flags:
        raise MatlabExtractionError("corrupt_or_unsupported_matlab")
    array_class = int(flags[0]) & 0xFF
    if array_class == _MX_OPAQUE:
        for data_type, value in elements[1:]:
            if data_type == _MI_MATRIX:
                _extract_matrix(
                    value,
                    endian,
                    budget,
                    parts,
                    depth=depth + 1,
                    inherited_label=inherited_label,
                )
        return
    dimensions = _dimensions(elements[1][0], elements[1][1], endian)
    array_name = _decode_name(elements[2][0], elements[2][1], endian).strip("\x00")
    label = array_name or inherited_label
    if label:
        budget.add_text(parts, label)

    content = elements[3:]
    if array_class == _MX_CHAR:
        if content:
            budget.add_text(
                parts,
                _decode_char_array(content[0][0], content[0][1], dimensions, endian),
            )
        return

    if array_class == _MX_STRUCT:
        if len(content) < 2:
            return
        names = _field_names(content[0], content[1], endian)
        for name in names:
            budget.add_text(parts, name)
        values = [element for element in content[2:] if element[0] == _MI_MATRIX]
        for index, (_, value) in enumerate(values):
            field_name = names[index % len(names)] if names else ""
            field_label = " ".join(part for part in (label, field_name) if part)
            _extract_matrix(
                value,
                endian,
                budget,
                parts,
                depth=depth + 1,
                inherited_label=field_label,
            )
        return

    if array_class in {_MX_CELL, _MX_OBJECT}:
        for data_type, value in content:
            if data_type == _MI_MATRIX:
                _extract_matrix(
                    value,
                    endian,
                    budget,
                    parts,
                    depth=depth + 1,
                    inherited_label=label,
                )
        return

    if array_class in {_MX_SPARSE, _MX_FUNCTION} or not content:
        return
    count = math.prod(dimensions) if dimensions else 0
    if (
        not _identity_related_label(label)
        or count <= 0
        or count > budget.max_numeric_values
    ):
        return
    values = _unpack_values(content[0][0], content[0][1], endian)
    if len(values) != count:
        return
    integers = []
    for value in values:
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            integers.append(str(int(numeric)))
    budget.add_text(parts, " ".join(integers))


def _extract_elements(
    payload: bytes | memoryview,
    endian: str,
    budget: _Budget,
    parts: list[str],
    *,
    depth: int,
) -> None:
    if depth > 32:
        raise MatlabExtractionError("matlab_resource_limit_exceeded")
    for data_type, value in _elements(payload, endian, budget):
        if data_type == _MI_MATRIX:
            _extract_matrix(value, endian, budget, parts, depth=depth + 1)
        elif data_type == _MI_COMPRESSED:
            decoded = _decompress(value, budget)
            _extract_elements(decoded, endian, budget, parts, depth=depth + 1)


def extract_matlab_text(
    payload: bytes,
    *,
    max_nodes: int,
    max_text_chars: int,
    max_decompressed_bytes: int,
    max_numeric_values: int,
) -> str:
    """Extract text from MATLAB v5/v6/v7 (non-HDF5) data without execution."""
    hdf5_file = payload.startswith(b"\x89HDF\r\n\x1a\n")
    if hdf5_file or b"MATLAB 7.3 MAT-file" in payload[:128]:
        raise MatlabExtractionError("matlab_v73_backend_unavailable")
    if len(payload) < 128 or not payload.startswith(b"MATLAB"):
        raise MatlabExtractionError("corrupt_or_unsupported_matlab")
    endian_marker = payload[126:128]
    if endian_marker == b"IM":
        endian = "<"
    elif endian_marker == b"MI":
        endian = ">"
    else:
        raise MatlabExtractionError("corrupt_or_unsupported_matlab")
    budget = _Budget(
        max_nodes=max_nodes,
        max_text_chars=max_text_chars,
        max_decompressed_bytes=max_decompressed_bytes,
        max_numeric_values=max_numeric_values,
    )
    parts: list[str] = []
    _extract_elements(payload[128:], endian, budget, parts, depth=0)
    return "\n".join(parts)
