import shutil
from function import *

shutil.rmtree(PROCESS_FOLDER)
os.mkdir(PROCESS_FOLDER)
shutil.rmtree(DIGITS_STR_FOLDER)
os.mkdir(DIGITS_STR_FOLDER)
shutil.rmtree(WRITE_FOLDER)
os.mkdir(WRITE_FOLDER)