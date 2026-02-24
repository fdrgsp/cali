"""Example of how to save labeled images."""

from cali.util import save_labeled_images

db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
out = "/Users/fdrgsp/Desktop/cali_test/labels"
save_labeled_images(db_path, out, overwrite=True)
