from cali.util import save_labeled_images

db_path = "manual_run.cali"
out = "/Users/fdrgsp/Desktop/cali_test/labels"
save_labeled_images(db_path, out, overwrite=True)
