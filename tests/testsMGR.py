import path
from src.data.PrepareInput import PrepareAudio as pa


class TestPrepareAudio():
    """Tests the PrepareAudio class in src/data
    """

    def test_mp3_file_accepted(self):
        """Test that a valid mp3 file is accepted
        """
        audio_prepper = pa()
        file = path.Path('tests/test_data/validfile_1.mp3')
        assert audio_prepper.start(file) is True
