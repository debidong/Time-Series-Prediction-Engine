# 文件的增删改查相关函数
import os

ALLOWED_EXTENSIONS = set(['csv'])
# 用于存储界面1的数据集
FILE_FOLDER = './dataset'
# 用于暂存界面3的数据集
TEMP_FOLDER = './temp'
# 用于存放界面2和界面3的结果
RESULT_FOLDER = './result'
# 用于存放界面2的训练模型
MODEL_PATH = './torch_models/'

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