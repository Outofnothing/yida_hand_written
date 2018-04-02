import shutil
from function import *

if __name__ == "__main__":
    shutil.rmtree(PROCESS_FOLDER)
    os.mkdir(PROCESS_FOLDER)
    shutil.rmtree(DIGITS_STR_FOLDER)
    os.mkdir(DIGITS_STR_FOLDER)
    shutil.rmtree(WRITE_FOLDER)
    os.mkdir(WRITE_FOLDER)
    for filename in os.listdir(READ_FOLDER):
        find_digits_str(filename)
    for filename in os.listdir(DIGITS_STR_FOLDER):
        digits_string = cv2.imread(os.path.join(DIGITS_STR_FOLDER, filename), 0)
        split_digits_str(digits_string, filename, is_vertical_writing(digits_string))
