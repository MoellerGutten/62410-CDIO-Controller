import os
import tempfile
import unittest
import shutil
from unittest.mock import patch, mock_open
from config import Config

class TestConfig(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'controller.config')
    
    def tearDown(self):
        # Clean up temp directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_loads_required_keys_ok(self):
        # Arrange
        config_content = "EV3_HOST=127.0.0.1\nEV3_PORT=1234\n"
        with open(self.config_path, 'w') as f:
            f.write(config_content)
        
        with patch('builtins.open', mock_open(read_data=config_content)), \
             patch('os.getcwd', return_value=self.test_dir):
            # Act
            c = Config()
            
            # Assert
            self.assertEqual(c.getStr("EV3_HOST"), "127.0.0.1")
            self.assertEqual(c.getNum("EV3_PORT"), 1234)
    
    def test_missing_required_key_raises(self):
        # Arrange
        config_content = "EV3_HOST=127.0.0.1\n"
        
        with patch('builtins.open', mock_open(read_data=config_content)):
            # Act / Assert
            with self.assertRaises(ValueError) as cm:
                Config()
            self.assertIn("Missing required config keys", str(cm.exception))
    
    def test_getStr_unknown_key_raises(self):
        config_content = "EV3_HOST=127.0.0.1\nEV3_PORT=1234\n"
        
        with patch('builtins.open', mock_open(read_data=config_content)):
            c = Config()
            
            with self.assertRaises(ValueError) as cm:
                c.getStr("UNKNOWN_KEY")
            self.assertIn("Key does not exist in config", str(cm.exception))
    
    def test_getNum_unknown_key_raises(self):
        config_content = "EV3_HOST=127.0.0.1\nEV3_PORT=1234\n"
        
        with patch('builtins.open', mock_open(read_data=config_content)):
            c = Config()
            
            with self.assertRaises(ValueError) as cm:
                c.getNum("UNKNOWN_KEY")
            self.assertIn("Key does not exist in config", str(cm.exception))
    
    def test_getNum_non_integer_value_raises(self):
        config_content = "EV3_HOST=127.0.0.1\nEV3_PORT=not_a_number\n"
        
        with patch('builtins.open', mock_open(read_data=config_content)):
            c = Config()
            
            with self.assertRaises(ValueError):
                c.getNum("EV3_PORT")

if __name__ == '__main__':
    unittest.main()
