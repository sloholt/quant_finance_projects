import unittest
import validation as v


class TestValidation(unittest.TestCase):

    def test_bad_date(self):
        with self.assertRaises(ValueError):
            v.validate_date("202613")

    def test_bad_factor(self):
        with self.assertRaises(ValueError):
            v.validate_factors(["RMRF", "XYZ"], ["RMRF", "SMB", "HML", "MOM"])


if __name__ == "__main__":
    unittest.main()
