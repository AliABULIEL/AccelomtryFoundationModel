"""
Efficient I/O operations for accelerometry data using HDF5 and Zarr.

Provides optimized storage with compression and chunking for large datasets.
"""
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any

import numpy as np
import h5py
import zarr
from numba import jit


def save_windows_hdf5(
    filepath: Union[str, Path],
    windows: np.ndarray,
    timestamps_start: np.ndarray,
    timestamps_end: np.ndarray,
    gap_flags: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = "gzip",
    compression_opts: int = 4,
    chunks: Optional[Tuple[int, ...]] = None
) -> None:
    """
    Save windowed accelerometry data to HDF5 format with compression.

    Args:
        filepath: Output HDF5 file path
        windows: Array of shape (N, C, win_n) containing windowed data
        timestamps_start: Array of start timestamps for each window
        timestamps_end: Array of end timestamps for each window
        gap_flags: Optional array of shape (N,) with gap flags per window
        metadata: Optional dictionary of metadata to store as attributes
        compression: Compression algorithm ("gzip", "lzf", "szip")
        compression_opts: Compression level (0-9 for gzip)
        chunks: Chunk shape for storage (default: auto-select based on data shape)

    Examples:
        >>> windows = np.random.randn(1000, 3, 819)
        >>> starts = pd.date_range('2020-01-01', periods=1000, freq='4.096s')
        >>> ends = starts + pd.Timedelta('8.192s')
        >>> save_windows_hdf5('output.h5', windows, starts.values, ends.values)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    N, C, win_n = windows.shape

    # Auto-select chunks if not provided
    # Good chunk size: ~1MB, balance between compression and random access
    if chunks is None:
        # Aim for ~1MB chunks
        samples_per_mb = 1024 * 1024 / (8 * C * win_n)  # 8 bytes per float64
        chunk_n = max(1, int(samples_per_mb))
        chunks = (min(chunk_n, N), C, win_n)

    with h5py.File(filepath, 'w') as f:
        # Store windowed data
        f.create_dataset(
            'windows',
            data=windows,
            dtype='float32',  # Use float32 to save space
            compression=compression,
            compression_opts=compression_opts,
            chunks=chunks
        )

        # Store timestamps (as int64 nanoseconds since epoch for efficiency)
        f.create_dataset(
            'timestamps_start',
            data=_datetime_to_ns(timestamps_start),
            dtype='int64',
            compression=compression,
            compression_opts=compression_opts
        )

        f.create_dataset(
            'timestamps_end',
            data=_datetime_to_ns(timestamps_end),
            dtype='int64',
            compression=compression,
            compression_opts=compression_opts
        )

        # Store gap flags if provided
        if gap_flags is not None:
            f.create_dataset(
                'gap_flags',
                data=gap_flags,
                dtype='uint8',
                compression=compression,
                compression_opts=compression_opts
            )

        # Store metadata as attributes
        if metadata is not None:
            for key, value in metadata.items():
                f.attrs[key] = value

        # Store shape information
        f.attrs['n_windows'] = N
        f.attrs['n_channels'] = C
        f.attrs['window_length'] = win_n


def load_windows_hdf5(
    filepath: Union[str, Path],
    load_timestamps: bool = True,
    load_gap_flags: bool = True,
    window_indices: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Load windowed accelerometry data from HDF5 format.

    Args:
        filepath: Input HDF5 file path
        load_timestamps: Whether to load timestamp arrays
        load_gap_flags: Whether to load gap flags
        window_indices: Optional array of window indices to load (for partial loading)

    Returns:
        Dictionary containing:
        - 'windows': Array of shape (N, C, win_n)
        - 'timestamps_start': Start timestamps (if load_timestamps=True)
        - 'timestamps_end': End timestamps (if load_timestamps=True)
        - 'gap_flags': Gap flags (if load_gap_flags=True and available)
        - 'metadata': Dictionary of stored metadata

    Examples:
        >>> data = load_windows_hdf5('output.h5')
        >>> data['windows'].shape
        (1000, 3, 819)
        >>> # Load subset
        >>> data = load_windows_hdf5('output.h5', window_indices=np.arange(100))
        >>> data['windows'].shape
        (100, 3, 819)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    result = {}

    with h5py.File(filepath, 'r') as f:
        # Load windows
        if window_indices is not None:
            result['windows'] = f['windows'][window_indices]
        else:
            result['windows'] = f['windows'][:]

        # Load timestamps
        if load_timestamps:
            if window_indices is not None:
                start_ns = f['timestamps_start'][window_indices]
                end_ns = f['timestamps_end'][window_indices]
            else:
                start_ns = f['timestamps_start'][:]
                end_ns = f['timestamps_end'][:]

            result['timestamps_start'] = _ns_to_datetime(start_ns)
            result['timestamps_end'] = _ns_to_datetime(end_ns)

        # Load gap flags
        if load_gap_flags and 'gap_flags' in f:
            if window_indices is not None:
                result['gap_flags'] = f['gap_flags'][window_indices]
            else:
                result['gap_flags'] = f['gap_flags'][:]

        # Load metadata
        result['metadata'] = dict(f.attrs)

    return result


def save_windows_zarr(
    filepath: Union[str, Path],
    windows: np.ndarray,
    timestamps_start: np.ndarray,
    timestamps_end: np.ndarray,
    gap_flags: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compressor: str = "zstd",
    compression_level: int = 3,
    chunks: Optional[Tuple[int, ...]] = None
) -> None:
    """
    Save windowed accelerometry data to Zarr format with compression.

    Zarr provides better parallel I/O and cloud storage support than HDF5.

    Args:
        filepath: Output Zarr directory path
        windows: Array of shape (N, C, win_n)
        timestamps_start: Start timestamps
        timestamps_end: End timestamps
        gap_flags: Optional gap flags
        metadata: Optional metadata dictionary
        compressor: Compression algorithm ("zstd", "blosc", "gzip")
        compression_level: Compression level (1-9)
        chunks: Chunk shape

    Examples:
        >>> windows = np.random.randn(1000, 3, 819)
        >>> starts = pd.date_range('2020-01-01', periods=1000, freq='4.096s')
        >>> ends = starts + pd.Timedelta('8.192s')
        >>> save_windows_zarr('output.zarr', windows, starts.values, ends.values)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    N, C, win_n = windows.shape

    # Select compressor
    if compressor == "zstd":
        comp = zarr.Blosc(cname='zstd', clevel=compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif compressor == "blosc":
        comp = zarr.Blosc(cname='lz4', clevel=compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif compressor == "gzip":
        comp = zarr.GZip(level=compression_level)
    else:
        raise ValueError(f"Unknown compressor: {compressor}")

    # Auto-select chunks if not provided
    if chunks is None:
        samples_per_mb = 1024 * 1024 / (4 * C * win_n)  # 4 bytes per float32
        chunk_n = max(1, int(samples_per_mb))
        chunks = (min(chunk_n, N), C, win_n)

    # Create Zarr store
    store = zarr.DirectoryStore(str(filepath))
    root = zarr.group(store=store, overwrite=True)

    # Store windows
    root.create_dataset(
        'windows',
        data=windows,
        dtype='float32',
        compressor=comp,
        chunks=chunks
    )

    # Store timestamps
    root.create_dataset(
        'timestamps_start',
        data=_datetime_to_ns(timestamps_start),
        dtype='int64',
        compressor=comp
    )

    root.create_dataset(
        'timestamps_end',
        data=_datetime_to_ns(timestamps_end),
        dtype='int64',
        compressor=comp
    )

    # Store gap flags
    if gap_flags is not None:
        root.create_dataset(
            'gap_flags',
            data=gap_flags,
            dtype='uint8',
            compressor=comp
        )

    # Store metadata
    root.attrs['n_windows'] = N
    root.attrs['n_channels'] = C
    root.attrs['window_length'] = win_n

    if metadata is not None:
        for key, value in metadata.items():
            root.attrs[key] = value


def load_windows_zarr(
    filepath: Union[str, Path],
    load_timestamps: bool = True,
    load_gap_flags: bool = True,
    window_indices: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Load windowed accelerometry data from Zarr format.

    Args:
        filepath: Input Zarr directory path
        load_timestamps: Whether to load timestamps
        load_gap_flags: Whether to load gap flags
        window_indices: Optional indices to load

    Returns:
        Dictionary with loaded data

    Examples:
        >>> data = load_windows_zarr('output.zarr')
        >>> data['windows'].shape
        (1000, 3, 819)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Directory not found: {filepath}")

    result = {}

    store = zarr.DirectoryStore(str(filepath))
    root = zarr.group(store=store)

    # Load windows
    if window_indices is not None:
        result['windows'] = root['windows'].oindex[window_indices]
    else:
        result['windows'] = root['windows'][:]

    # Load timestamps
    if load_timestamps:
        if window_indices is not None:
            start_ns = root['timestamps_start'].oindex[window_indices]
            end_ns = root['timestamps_end'].oindex[window_indices]
        else:
            start_ns = root['timestamps_start'][:]
            end_ns = root['timestamps_end'][:]

        result['timestamps_start'] = _ns_to_datetime(start_ns)
        result['timestamps_end'] = _ns_to_datetime(end_ns)

    # Load gap flags
    if load_gap_flags and 'gap_flags' in root:
        if window_indices is not None:
            result['gap_flags'] = root['gap_flags'].oindex[window_indices]
        else:
            result['gap_flags'] = root['gap_flags'][:]

    # Load metadata
    result['metadata'] = dict(root.attrs)

    return result


def _datetime_to_ns(timestamps: np.ndarray) -> np.ndarray:
    """Convert datetime64 to int64 nanoseconds since epoch."""
    if np.issubdtype(timestamps.dtype, np.datetime64):
        return timestamps.astype('datetime64[ns]').astype('int64')
    else:
        # Already in nanoseconds
        return timestamps.astype('int64')


def _ns_to_datetime(timestamps_ns: np.ndarray) -> np.ndarray:
    """Convert int64 nanoseconds to datetime64."""
    return timestamps_ns.astype('datetime64[ns]')
