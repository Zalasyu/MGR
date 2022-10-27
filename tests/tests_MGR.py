from src.data.PrepareInput import PrepareAudio as pa


class TestPrepareAudio():
    """Tests the PrepareAudio class in src/data."""
    pa = pa()

    def test_mp3_file_accepted(self):
        """Test that a valid .mp3 file is accepted"""

        file_path = 'tests/test_data/validfile_1.mp3'
        assert self.pa.check_file_type(file_path) is True

    def test_mp4_file_accepted(self):
        """Test that a valid .mp4 file is accepted"""

        file_path = 'tests/test_data/validfile_4.mp4'
        assert self.pa.check_file_type(file_path) is True

    def test_wav_file_accepted(self):
        """Test that a valid .wav file is accepted"""

        file_path = 'tests/test_data/validfile_2.wav'
        assert self.pa.check_file_type(file_path) is True

    def test_au_file_accepted(self):
        """Test that a valid .au file is accepted"""

        file_path = 'tests/test_data/validfile_3.au'
        assert self.pa.check_file_type(file_path) is True

    def test__file_not_accepted(self):
        """Test that a valid .au file is accepted"""

        file_path = 'tests/test_data/invalidfile_1.pdf'
        assert self.pa.check_file_type(file_path) is False

    def test_file_exists(self):
        """Test if an existing file is detected."""
        file_path = 'tests/test_data/validfile_1.mp3'
        assert self.pa.check_file_exists(file_path) is True

    def test_file_doesnt_exists(self):
        """Test if an non-existing file is not detected."""
        file_path = 'tests/test_data/nonexistingFile.mp3'
        assert self.pa.check_file_exists(file_path) is False
