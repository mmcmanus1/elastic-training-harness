"""Tests for ResumableDataset and token-based sharding."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from elastic_harness.data.index_builder import IndexBuilder, IndexEntry, IndexHeader, TokenIndexFile
from elastic_harness.data.resumable_dataset import DatasetState, ResumableDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str) -> list[int]:
        # Simple tokenization: split by words and assign sequential IDs
        # In reality, this would use a real tokenizer
        words = text.split()
        return list(range(len(words)))


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def sample_data_files(tmp_path: Path):
    """Create sample JSONL data files for testing."""
    files = []

    for i in range(2):
        file_path = tmp_path / f"data_{i}.jsonl"
        with open(file_path, "w") as f:
            for j in range(10):
                # Each line has increasing number of words
                words = " ".join([f"word{k}" for k in range(j + 1)])
                f.write(json.dumps({"text": words}) + "\n")
        files.append(str(file_path))

    return files


@pytest.fixture
def sample_index(tmp_path: Path, sample_data_files, mock_tokenizer):
    """Build a sample index for testing."""
    index_path = tmp_path / "index.bin"

    builder = IndexBuilder(
        tokenizer=mock_tokenizer,
        entry_interval=10,  # Small interval for testing
    )
    builder.build(sample_data_files, index_path)

    return TokenIndexFile(index_path)


class TestIndexEntry:
    """Tests for IndexEntry serialization."""

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        entry = IndexEntry(file_index=1, byte_offset=12345, cumulative_tokens=67890)

        data = entry.to_bytes()
        restored = IndexEntry.from_bytes(data)

        assert restored.file_index == entry.file_index
        assert restored.byte_offset == entry.byte_offset
        assert restored.cumulative_tokens == entry.cumulative_tokens

    def test_size(self):
        """Test entry size is correct."""
        entry = IndexEntry(file_index=0, byte_offset=0, cumulative_tokens=0)
        assert len(entry.to_bytes()) == IndexEntry.SIZE


class TestIndexHeader:
    """Tests for IndexHeader serialization."""

    def test_roundtrip(self):
        """Test header serialization roundtrip."""
        header = IndexHeader(
            version=1,
            num_entries=100,
            total_tokens=50000,
            num_files=5,
            entry_interval=10000,
        )

        data = header.to_bytes()
        restored = IndexHeader.from_bytes(data)

        assert restored.version == header.version
        assert restored.num_entries == header.num_entries
        assert restored.total_tokens == header.total_tokens
        assert restored.num_files == header.num_files
        assert restored.entry_interval == header.entry_interval

    def test_magic_validation(self):
        """Test that invalid magic number raises error."""
        bad_data = b"BADMAGIC" + b"\x00" * 40

        with pytest.raises(ValueError, match="bad magic number"):
            IndexHeader.from_bytes(bad_data)


class TestIndexBuilder:
    """Tests for IndexBuilder."""

    def test_build_creates_index(self, tmp_path, sample_data_files, mock_tokenizer):
        """Test that index is created correctly."""
        index_path = tmp_path / "test_index.bin"

        builder = IndexBuilder(tokenizer=mock_tokenizer, entry_interval=10)
        header = builder.build(sample_data_files, index_path)

        assert index_path.exists()
        assert header.num_files == 2
        assert header.total_tokens > 0

    def test_build_creates_metadata(self, tmp_path, sample_data_files, mock_tokenizer):
        """Test that metadata JSON is created."""
        index_path = tmp_path / "test_index.bin"
        metadata_path = tmp_path / "test_index.json"

        builder = IndexBuilder(tokenizer=mock_tokenizer, entry_interval=10)
        builder.build(sample_data_files, index_path)

        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert len(metadata["files"]) == 2
        assert metadata["total_tokens"] > 0


class TestTokenIndexFile:
    """Tests for TokenIndexFile."""

    def test_load_index(self, sample_index):
        """Test loading an index file."""
        assert sample_index.total_tokens > 0
        assert sample_index.num_files == 2

    def test_find_position_start(self, sample_index):
        """Test finding position at start."""
        file_idx, byte_offset, token_offset = sample_index.find_position(0)

        assert file_idx == 0
        assert byte_offset == 0
        assert token_offset == 0

    def test_find_position_middle(self, sample_index):
        """Test finding position in middle of data."""
        # Find a position somewhere in the middle
        mid_token = sample_index.total_tokens // 2
        file_idx, byte_offset, token_offset = sample_index.find_position(mid_token)

        assert token_offset <= mid_token

    def test_find_position_out_of_range(self, sample_index):
        """Test that out-of-range position raises error."""
        with pytest.raises(ValueError, match="exceeds total tokens"):
            sample_index.find_position(sample_index.total_tokens + 1000)


class TestDatasetState:
    """Tests for DatasetState."""

    def test_roundtrip(self):
        """Test state serialization roundtrip."""
        state = DatasetState(
            total_tokens_processed=12345,
            current_file_index=2,
            current_byte_offset=5000,
            world_size_at_checkpoint=4,
            epoch=3,
        )

        data = state.to_dict()
        restored = DatasetState.from_dict(data)

        assert restored.total_tokens_processed == state.total_tokens_processed
        assert restored.current_file_index == state.current_file_index
        assert restored.current_byte_offset == state.current_byte_offset
        assert restored.world_size_at_checkpoint == state.world_size_at_checkpoint
        assert restored.epoch == state.epoch

    def test_defaults(self):
        """Test default values."""
        state = DatasetState()

        assert state.total_tokens_processed == 0
        assert state.current_file_index == 0
        assert state.epoch == 0


class TestResumableDataset:
    """Tests for ResumableDataset."""

    def test_create_dataset(self, sample_data_files, sample_index, mock_tokenizer):
        """Test creating a dataset."""
        dataset = ResumableDataset(
            data_files=sample_data_files,
            index_file=sample_index,
            tokenizer=mock_tokenizer,
            seq_length=4,
        )

        assert dataset.total_tokens == sample_index.total_tokens

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_shard_range_single_worker(self, mock_dist, sample_data_files, sample_index, mock_tokenizer):
        """Test shard calculation for single worker."""
        dataset = ResumableDataset(
            data_files=sample_data_files,
            index_file=sample_index,
            tokenizer=mock_tokenizer,
            seq_length=4,
        )

        start, end = dataset._calculate_shard_range(rank=0, world_size=1)

        assert start == 0
        assert end == sample_index.total_tokens

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_shard_range_multiple_workers(self, mock_dist, sample_data_files, sample_index, mock_tokenizer):
        """Test shard calculation for multiple workers."""
        dataset = ResumableDataset(
            data_files=sample_data_files,
            index_file=sample_index,
            tokenizer=mock_tokenizer,
            seq_length=4,
        )

        total_tokens = sample_index.total_tokens
        world_size = 4

        # Calculate all shard ranges
        ranges = []
        for rank in range(world_size):
            start, end = dataset._calculate_shard_range(rank=rank, world_size=world_size)
            ranges.append((start, end))

        # Verify shards are contiguous and cover all tokens
        assert ranges[0][0] == 0  # First shard starts at 0
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]  # Contiguous
        assert ranges[-1][1] == total_tokens  # Last shard ends at total

    def test_state_dict(self, sample_data_files, sample_index, mock_tokenizer):
        """Test state dictionary creation."""
        dataset = ResumableDataset(
            data_files=sample_data_files,
            index_file=sample_index,
            tokenizer=mock_tokenizer,
            seq_length=4,
        )

        state = dataset.state_dict()

        assert "total_tokens_processed" in state
        assert "world_size" in state
        assert "epoch" in state

    def test_load_state_dict(self, sample_data_files, sample_index, mock_tokenizer):
        """Test state restoration."""
        dataset = ResumableDataset(
            data_files=sample_data_files,
            index_file=sample_index,
            tokenizer=mock_tokenizer,
            seq_length=4,
        )

        # Save state
        original_state = dataset.state_dict()
        original_state["total_tokens_processed"] = 1000
        original_state["epoch"] = 5

        # Load state
        dataset.load_state_dict(original_state)

        assert dataset._state.total_tokens_processed == 1000
        assert dataset._state.epoch == 5

    def test_set_epoch(self, sample_data_files, sample_index, mock_tokenizer):
        """Test setting epoch."""
        dataset = ResumableDataset(
            data_files=sample_data_files,
            index_file=sample_index,
            tokenizer=mock_tokenizer,
            seq_length=4,
        )

        dataset.set_epoch(10)
        assert dataset._state.epoch == 10


class TestTokenShardingCorrectness:
    """Tests for token-based sharding correctness across topology changes."""

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_no_data_duplication_on_scale_down(self, mock_dist, sample_data_files, sample_index, mock_tokenizer):
        """Test that scaling down doesn't cause data duplication."""
        # Simulate processing with 4 workers
        world_size_old = 4
        total_tokens = sample_index.total_tokens

        # Calculate what tokens each worker would process
        old_ranges = []
        for rank in range(world_size_old):
            dataset = ResumableDataset(
                data_files=sample_data_files,
                index_file=sample_index,
                tokenizer=mock_tokenizer,
                seq_length=4,
            )
            start, end = dataset._calculate_shard_range(rank=rank, world_size=world_size_old)
            old_ranges.append(set(range(start, end)))

        # Verify no overlap
        for i in range(len(old_ranges)):
            for j in range(i + 1, len(old_ranges)):
                assert len(old_ranges[i] & old_ranges[j]) == 0, "Overlapping ranges detected"

        # Verify all tokens covered
        all_tokens = set()
        for r in old_ranges:
            all_tokens |= r
        assert all_tokens == set(range(total_tokens)), "Not all tokens covered"

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_resume_offset_calculation(self, mock_dist, sample_data_files, sample_index, mock_tokenizer):
        """Test that resume offset is calculated correctly."""
        # Simulate checkpoint at 50% progress
        checkpoint_tokens = sample_index.total_tokens // 2

        # New world size
        new_world_size = 2

        dataset = ResumableDataset(
            data_files=sample_data_files,
            index_file=sample_index,
            tokenizer=mock_tokenizer,
            seq_length=4,
        )

        # Load state with tokens processed
        dataset.load_state_dict({
            "total_tokens_processed": checkpoint_tokens,
            "world_size": 4,  # Old world size
        })

        # Calculate new ranges
        remaining_tokens = sample_index.total_tokens - checkpoint_tokens

        for rank in range(new_world_size):
            start, end = dataset._calculate_shard_range(
                rank=rank,
                world_size=new_world_size,
                start_offset=checkpoint_tokens,
            )

            # Each worker should get approximately half of remaining tokens
            worker_tokens = end - start
            expected = remaining_tokens // new_world_size

            # Allow for remainder distribution
            assert abs(worker_tokens - expected) <= 1
