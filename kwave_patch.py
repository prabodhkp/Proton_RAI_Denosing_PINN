"""
Patch k-Wave to skip binary installation in Streamlit Cloud.
Must be imported before kwave.
"""
import sys
import os

# Patch os.makedirs and urllib.request.urlretrieve before kwave loads
_original_makedirs = os.makedirs
_skip_kwave_bin = False

def _patched_makedirs(name, *args, **kwargs):
    global _skip_kwave_bin
    if 'kwave' in str(name) and 'bin' in str(name):
        _skip_kwave_bin = True
        return  # Skip kwave bin directory creation
    return _original_makedirs(name, *args, **kwargs)

os.makedirs = _patched_makedirs

# Patch urlretrieve to skip if makedirs was skipped
import urllib.request
_original_urlretrieve = urllib.request.urlretrieve

def _patched_urlretrieve(url, filename=None, *args, **kwargs):
    global _skip_kwave_bin
    if _skip_kwave_bin and 'kwave' in str(filename):
        _skip_kwave_bin = False
        return filename, None  # Return dummy response
    return _original_urlretrieve(url, filename, *args, **kwargs)

urllib.request.urlretrieve = _patched_urlretrieve

# Patch open() to skip hash file reads
_original_open = open
_dummy_hash_used = False

def _patched_open(file, mode='r', *args, **kwargs):
    global _skip_kwave_bin, _dummy_hash_used
    if _skip_kwave_bin and 'kwave' in str(file) and mode == 'rb':
        _dummy_hash_used = True
        # Return a fake file object that produces a dummy hash
        import io
        return io.BytesIO(b'dummy')
    if _dummy_hash_used and 'metadata.json' in str(file):
        _dummy_hash_used = False
        _skip_kwave_bin = False
    return _original_open(file, mode, *args, **kwargs)

# Replace built-in open
import builtins
builtins.open = _patched_open

print("✓ k-Wave patched for Streamlit Cloud")
