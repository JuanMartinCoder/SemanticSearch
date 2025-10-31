import unittest


from text_processing import (
    preprocess_text,
    tokenize,
)

class TestTextProcessing(unittest.TestCase):
    def test_remove_punctuation(self):
        text = "This is a test. It's a good one."
        expected = "this is a test its a good one"
        self.assertEqual(preprocess_text(text), expected)

    def test_tokenize(self):
        text = "The Chronicles of Narnia: Prince Caspian"
        expected = ["chronicles", "narnia", "prince", "caspian"]
        self.assertEqual(tokenize(text), expected)



if __name__ == "__main__":
    unittest.main()