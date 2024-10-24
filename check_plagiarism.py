import os
import argparse
import shutil


def get_meta(root):
    meta_info = {
        'a1': {
            'downloads': f'{root}/a1/downloads/COMP7404_1A_2024-Submit a1-3484224.zip',
            'naive_unzip': f'{root}/a1/downloads/naive_unzip',
            'standard_unzip': f'{root}/a1/downloads/standard_unzip',
            'submissions': f'{root}/a1/submissions/',
            'resubmit': f'{root}/a1/resubmit_after_ddl/',
            'base': f'{root}/a1/base/',
            'checked': f'{root}/a1/checked/'
        }
    }
    return meta_info


def get_base_files(base_path: str):
    base_files = [x for x in os.listdir(base_path) if '.py' in x]
    base_files = [os.path.join(base_path, x) for x in base_files]
    return base_files


def get_checked_dirs(submission_path, resubmit_path):
    first_dirs = os.listdir(submission_path)
    second_dirs = os.listdir(resubmit_path)
    merged_dirs = []
    for i in range(len(first_dirs)):
        student_folder = first_dirs[i]
        if 'assign' not in student_folder:
            continue
        if student_folder in second_dirs:
            merged_dirs.append(os.path.join(resubmit_path, student_folder))
        else:
            merged_dirs.append(os.path.join(submission_path, student_folder))
    return merged_dirs


def copy_checked_files(checked_dirs, checked_folder):
    ''' Rename and copy to checked/ '''
    copied_dirs = []
    cnt = len(checked_dirs)
    for i in range(cnt):
        abs_student_dir = checked_dirs[i]
        _, student_folder = os.path.split(abs_student_dir)
        abs_copied_dir = os.path.join(checked_folder, student_folder.replace(' ', '_'))
        shutil.copytree(
            abs_student_dir, 
            abs_copied_dir,
            dirs_exist_ok=True
        )
        copied_dirs.append(abs_copied_dir)
    return copied_dirs



def run_moss(root, assignment, moss_path):
    os_cmd = f"perl {moss_path} -l python "
    meta_data = get_meta(root)[assignment]
    base_path = meta_data['base']
    submission_path = meta_data['submissions']
    resubmit_path = meta_data['resubmit']
    checked_path = meta_data['checked']

    # 1. Base files from given example code
    base_files = get_base_files(base_path)
    base_cmd = [f"-b {x} " for x in base_files]
    os_cmd += " ".join(base_cmd)
    print(os_cmd)

    # 2. Being checked folders
    checked_dirs = get_checked_dirs(submission_path, resubmit_path)
    checked_dirs.sort()

    # 3. Rename and copy to `checked/`
    checked_dirs = copy_checked_files(checked_dirs, checked_path)

    # 4. Concat as command
    cat_dirs: str = " ".join([f"{d}/*.py " for d in checked_dirs])
    os_cmd += f"-d {cat_dirs} "

    # 5. Submit to MOSS server
    print(os_cmd)
    os.system(os_cmd)


if __name__ == "__main__":
    args = argparse.ArgumentParser("Checking plagiarism")
    args.add_argument("-r", "--root", type=str, default='D:/Documents/HKU/COMP7404_Marking/',
        help='Where you place your "a1/submissions/", "a1/eval_env/", and "a1/resubmit_after_ddl/" ')
    args.add_argument("-a", "--assignment", type=str, default="a1")
    args.add_argument("--moss_path", type=str, default="moss.pl",
                      help="The MOSS script provided by: https://theory.stanford.edu/~aiken/moss/")
    args = args.parse_args()
    
    run_moss(
        args.root,
        args.assignment,
        args.moss_path,
    )
