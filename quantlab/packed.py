"""QLB1: bounded row-major symmetric int8 weights, explicit IEEE scales, CRC32.

A small educational interchange format, not ONNX/GGUF or a secure container.
No pickle, code execution, strings, paths, compression or dynamic allocation.
"""
from __future__ import annotations
import math
import struct
import zlib

HEADER = struct.Struct('<4sBBBBIIII')
MAX_ROWS, MAX_COLS = 8, 128
MAX_BYTES = HEADER.size + 8 * MAX_ROWS + MAX_ROWS * MAX_COLS + 4


def _dimension(n, low, high):
    if type(n) is not int or not low <= n <= high:
        raise ValueError('dimension outside format limits')
    return n


def validate_quantized(packed):
    if type(packed) is not dict or set(packed) != {'mode', 'q', 'scales'}:
        raise ValueError('expected mode, q and scales only')
    if packed['mode'] not in ('per-tensor', 'per-channel'):
        raise ValueError('unknown quantization mode')
    q, scales = packed['q'], packed['scales']
    if type(q) is not list:
        raise ValueError('q must be a matrix')
    rows = _dimension(len(q), 2, MAX_ROWS)
    if type(q[0]) is not list:
        raise ValueError('q rows must be lists')
    cols = _dimension(len(q[0]), 1, MAX_COLS)
    for row in q:
        if type(row) is not list or len(row) != cols or any(type(v) is not int or not -127 <= v <= 127 for v in row):
            raise ValueError('rectangular symmetric int8 matrix required; -128 is reserved')
    n = 1 if packed['mode'] == 'per-tensor' else rows
    if type(scales) is not list or len(scales) != n:
        raise ValueError('wrong scale count')
    if any(type(s) not in (int, float) or not 0 < s <= 1e6 or not math.isfinite(s) for s in scales):
        raise ValueError('scales must be positive finite values at most 1e6')
    return rows, cols


def encode(packed, scale_bits=64):
    """Keep original int8 codes; round only scale metadata to its storage type."""
    rows, cols = validate_quantized(packed)
    if type(scale_bits) is not int or scale_bits not in (32, 64):
        raise ValueError('scale_bits must be 32 or 64')
    code = 'f' if scale_bits == 32 else 'd'
    scale_blob = struct.pack('<' + code * len(packed['scales']), *packed['scales'])
    restored = struct.unpack('<' + code * len(packed['scales']), scale_blob)
    if any(not math.isfinite(s) or s <= 0 for s in restored):
        raise ValueError('scale loses positive finite representation')
    head = HEADER.pack(b'QLB1', 1, int(packed['mode'] == 'per-channel'), scale_bits // 8, 0,
                       rows, cols, rows * cols, len(restored))
    body = head + scale_blob + bytes(v & 255 for row in packed['q'] for v in row)
    return body + struct.pack('<I', zlib.crc32(body))


def decode(blob):
    if type(blob) not in (bytes, bytearray) or not HEADER.size + 4 <= len(blob) <= MAX_BYTES:
        raise ValueError('invalid QLB1 byte length')
    magic, version, mode, size, flags, rows, cols, count, ns = HEADER.unpack_from(blob)
    if magic != b'QLB1' or version != 1 or mode not in (0, 1) or size not in (4, 8) or flags:
        raise ValueError('unsupported header')
    _dimension(rows, 2, MAX_ROWS)
    _dimension(cols, 1, MAX_COLS)
    if count != rows * cols or ns != (rows if mode else 1):
        raise ValueError('inconsistent counts')
    if len(blob) != HEADER.size + ns * size + count + 4:
        raise ValueError('truncation or trailing bytes')
    if zlib.crc32(blob[:-4]) != struct.unpack_from('<I', blob, len(blob) - 4)[0]:
        raise ValueError('CRC32 mismatch (accidental corruption check, not authentication)')
    scales = list(struct.unpack_from('<' + ('f' if size == 4 else 'd') * ns, blob, HEADER.size))
    start = HEADER.size + ns * size
    signed = [v if v < 128 else v - 256 for v in blob[start:-4]]
    packed = {'mode': 'per-channel' if mode else 'per-tensor', 'scales': scales,
              'q': [signed[i * cols:(i + 1) * cols] for i in range(rows)]}
    validate_quantized(packed)
    return {'scale_bits': size * 8, 'rows': rows, 'cols': cols, 'packed': packed,
            'layout': {'header_bytes': HEADER.size, 'scale_bytes': ns * size,
                       'int8_bytes': count, 'checksum_bytes': 4, 'total_bytes': len(blob)}}
