# 文件的增删改查相关函数
import os

ALLOWED_EXTENSIONS = set(['csv'])
FILE_FOLDER = './upload'
TEMP_FOLDER = './temp'
RESULT_FOLDER = './result'
current_file_path = './upload'
current_file_row = 0
current_file_column = 0
current_file_name = ''

def is_allowed_file(filename: str) -> bool:
    """根据文件名判断文件类型是否合法"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def is_duplicate_name(current_file_path, directory_path=FILE_FOLDER) -> bool:
    """检查文件名是否冗余"""
    current_file_name = os.path.basename(current_file_path)
    duplicate_file_path = os.path.join(directory_path, current_file_name)
    if os.path.exists(duplicate_file_path):
        print(f'Duplicate file found: {duplicate_file_path}')
        return True
    else:
        print('No duplicate file found.')
        return False