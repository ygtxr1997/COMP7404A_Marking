import argparse
import os
import zipfile
from tqdm import tqdm
import shutil


root = 'D:/Documents/HKU/COMP7404_Marking/'
# root = '/Users/greensongavin/Documents/HKU/COMP7404A/'
meta_info = {
    'a1': {
        'downloads': f'{root}/a1/downloads/COMP7404_1A_2024-Submit a1-3484224.zip',
        'naive_unzip': f'{root}/a1/downloads/naive_unzip',
        'standard_unzip': f'{root}/a1/downloads/standard_unzip',
        'submissions': f'{root}/a1/submissions/',
        'resubmit': f'{root}/a1/resubmit_after_ddl/',
    },
    'a2': {
        'downloads': f'{root}/a2/downloads/COMP7404_1A_2024-Submit a2-3486306.zip',
        'naive_unzip': f'{root}/a2/downloads/naive_unzip',
        'standard_unzip': f'{root}/a2/downloads/standard_unzip',
        'submissions': f'{root}/a2/submissions/',
        'resubmit': f'{root}/a2/resubmit_after_ddl/',
    },
    'a3': {
        'downloads': f'{root}/a3/downloads/COMP7404_1A_2024-Submit a3-3507151.zip',
        'naive_unzip': f'{root}/a3/downloads/naive_unzip',
        'standard_unzip': f'{root}/a3/downloads/standard_unzip',
        'submissions': f'{root}/a3/submissions/',
        'resubmit': f'{root}/a3/resubmit_after_ddl/',
    }
}


def unzip_file(zip_path, extract_to=None, verbose=False, iterative=False) -> str:
    if extract_to is None:
        extract_to = os.path.splitext(zip_path)[0]
    if os.path.exists(extract_to):
        print(f'{extract_to} already exists, skipping.')
        return extract_to
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    if verbose:
        print(f'{zip_path} extracted to {extract_to}.')
    if (iterative):
        # checking unzip files
        out_files = os.listdir(extract_to)
        for out_file in out_files:
            if os.path.splitext(out_file) in ('zip', '7z'):
                unzip_file(
                    os.path.join(extract_to, out_file),
                    os.path.join(extract_to, os.path.join(extract_to, os.path.splitext(out_file))),
                    iterative=True,
                )
    return extract_to


def iterative_find_files(folder, target='p1.py'):
    fn_list = os.listdir(folder)
    if target in fn_list:
        return folder
    for fn in fn_list:
        fn_abs = os.path.join(folder, fn)
        if os.path.isdir(fn_abs):
            res = iterative_find_files(fn_abs, target)
            if res != None:
                return res
    return None


def copy_to_standard_dir(unzip_dir, naive_dir, standard_dir):
    name_id_suffix_folders = os.listdir(unzip_dir)
    name_id_suffix_folders = [x for x in name_id_suffix_folders if '.DS_Store' not in x]
    for name_id_suffix in tqdm(name_id_suffix_folders):
        print(name_id_suffix)
        submitted_files = os.listdir(os.path.join(unzip_dir, name_id_suffix))
        if len(submitted_files) > 1: 
            print(f'[Warning] More than one files found in {name_id_suffix}')
            for submitted_zip in submitted_files:
                if '.zip' in submitted_zip:
                    break
        else:
            submitted_zip = submitted_files[0]
        extracted_path = unzip_file(
            os.path.join(unzip_dir, name_id_suffix, submitted_zip),
            os.path.join(naive_dir, name_id_suffix),
            iterative=False,
        )
        # iterative find the `p1.py`
        p1_path = iterative_find_files(extracted_path)

        # copy all files under dir containing `p1.py` to standard_dir
        shutil.copytree(f'{p1_path}', f'{standard_dir}/{name_id_suffix}/',
            dirs_exist_ok=True,
        )


def copy_resubmit_to_submissions(resubmit, submissions):
    resub_fns = os.listdir(resubmit)
    initial_fns = os.listdir(submissions)
    for resub in resub_fns:
        if resub in initial_fns:
            print(f'Copy resubmitted file {resub} to submissions/')
            shutil.copytree(
                os.path.join(resubmit, resub),
                os.path.join(submissions, resub),
                dirs_exist_ok=True
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser('Preprocess')
    args.add_argument('-a', '--assignment', type=str, choices=['a1', 'a2', 'a3'])
    opts = args.parse_args()

    meta_data = meta_info[opts.assignment]

    unzip_dir = unzip_file(meta_data['downloads'], verbose=True)
    copy_to_standard_dir(unzip_dir, meta_data['naive_unzip'], meta_data['standard_unzip'])

    # Copy to submissions/ folder
    shutil.copytree(
        meta_data['standard_unzip'], 
        meta_data['submissions'],
        dirs_exist_ok=True
    )

    # Do NOT copy, we now load resubmitted files in grader_ta_a*.py
    # # Use resubmitted files to cover moodle files
    # copy_resubmit_to_submissions(meta_data['resubmit'], meta_data['submissions'])
