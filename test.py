import unittest
from automl import get_data_from_kaggle, automl_pycaret, submit_kaggle
import os

class TestDataRetrieval(unittest.TestCase):
    def test_get_data_from_kaggle(self):
        get_data_from_kaggle()
        self.assertTrue(os.path.exists('Data/train.csv'))
        self.assertTrue(os.path.exists('Data/test.csv'))

class TestAutoMLPyCaret(unittest.TestCase):
    def test_automl_pycaret(self):
        model_path = automl_pycaret()
        self.assertTrue(os.path.exists(model_path))

class TestSubmitKaggle(unittest.TestCase):
    def test_submit_kaggle(self):
        try:
            submit_kaggle()
        except Exception as e:
            self.fail(f'submit_kaggle raised {e.__class__.__name__} unexpectedly!')

if __name__ == '__main__':
    unittest.main()
