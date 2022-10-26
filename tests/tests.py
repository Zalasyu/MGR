# Dependencies
import os
import unittest
from ..Teststuf.src.data.PrepareInput import PrepareAudio


class PrepareAudioTestCase(unittest.TestCase):
    """Contains various tests intended to be used with
    ...the PrepareAudio class."""

    pa = PrepareAudio()

    def test1(self):
        """Check if .mp3 file is accepted."""
        input_file = 'TestInputFiles/validfile_1.mp3'
        self.assertEqual(self.pa.start(input_file), True)

    def test2(self):
        """Check if .wav file is accepted."""
        input_file = 'TestInputFiles/validfile_2.wav'
        self.assertEqual(self.pa.start(input_file), True)

    def test3(self):
        """Check if .au file is accepted."""
        input_file = 'TestInputFiles/validfile_3.au'
        self.assertEqual(self.pa.start(input_file), True)

    def test4(self):
        """Check if .mp4 file is accepted."""
        input_file = 'TestInputFiles/validfile_4.mp4'
        self.assertEqual(self.pa.start(input_file), True)

    def test5(self):
        """Check if .mp3 file successfully converts to .wav"""
        converted_file = 'Inputs/validfile_1.wav'
        self.assertEqual(os.path.exists(converted_file), True)

    def test6(self):
        """Check if .wav file successfully converts to .wav"""
        converted_file = 'Inputs/validfile_2.wav'
        self.assertEqual(os.path.exists(converted_file), True)

    def test7(self):
        """Check if .au file successfully converts to .wav"""
        converted_file = 'Inputs/validfile_3.wav'
        self.assertEqual(os.path.exists(converted_file), True)

    def test8(self):
        """Check if .mp4 file successfully converts to .wav"""
        converted_file = 'Inputs/validfile_4.wav'
        self.assertEqual(os.path.exists(converted_file), True)

    def test9(self):
        """Check if a mel spec is generated from a .mp3"""
        img_file = 'Outputs/validfile_1_ms.png'
        self.assertEqual(os.path.exists(img_file), True)

    def test10(self):
        """Check if a mel spec is generated from a .wav"""
        img_file = 'Outputs/validfile_2_ms.png'
        self.assertEqual(os.path.exists(img_file), True)

    def test11(self):
        """Check if a mel spec is generated from a .au"""
        img_file = 'Outputs/validfile_3_ms.png'
        self.assertEqual(os.path.exists(img_file), True)

    def test12(self):
        """Check if a mel spec is generated from a .mp4"""
        img_file = 'Outputs/validfile_4_ms.png'
        self.assertEqual(os.path.exists(img_file), True)

    def test13(self):
        """Check if a .pdf is not accepted"""
        pdf_file = 'TestInputFiles/invalidfile_1.pdf'
        self.assertEqual(self.pa.start(pdf_file), False)

    def test14(self):
        """Check if a .png is not accepted"""
        img_file = 'TestInputFiles/invalidfile_2.png'
        self.assertEqual(self.pa.start(img_file), False)

    def test15(self):
        """Check if a an invalid path returns False"""
        path = 'invalid/path/to/an/audio/file.mp3'
        self.assertEqual(self.pa.start(path), False)


if __name__ == '__main__':
    unittest.main()
