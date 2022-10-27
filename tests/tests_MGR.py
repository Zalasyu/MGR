# Import path library
from pathlib import Path

from src.data.PrepareInput import PrepareAudio as pa


class TestPrepareAudio():
    """Tests the PrepareAudio class in src/data
    """
    pa = pa()

    def test_mp3_file_accepted(self):
        """Test that a valid mp3 file is accepted"""
        file_path = 'tests/test_data/validfile_1.mp3'
        assert self.pa.check_file_exists(file_path) is True
