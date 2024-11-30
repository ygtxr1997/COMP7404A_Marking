#Do not make changes to this file
import os, difflib, sys
import time, argparse, threading, importlib, multiprocessing, shutil
import logging
import functools
from typing import List, Dict

from grader_ta_a1 import SuprressPrint, TimeoutHelper, init_logger
from grader_ta_a1 import make_abs_path, read_emails, read_discount
from grader_ta_a1 import calc_assignment_score


verbose = False  # some students are using this value
sup_printer = SuprressPrint("")


# Customized for different assignments
def grade(problem_id, test_case_id, student_code_problem, student_code_parse,
        assignment_id: str = 'a3',
        verbose: bool = False,
        timeout_limit: float = 5 * 60 + 1,  # max running time for each testing case is 5 min
        student_folder: str = "",
        is_resubmit: bool = False,
    ) -> dict:
    if verbose:
        print('Grading Problem', problem_id,':')
    logger.info(f'Running: Problem={problem_id}, is_resubmit={is_resubmit}')
    if test_case_id > 0:
        # Single test case
        check_test_case(problem_id, test_case_id, student_code_problem, student_code_parse)
    else:
        # Multiple test cases
        num_test_cases = test_case_id * (-1)

        # 1. Vanilla testing mode
        passed_visible_cases, failed_visible_cases = [], []  # visible testing cases
        for i in range(1, num_test_cases+1):
            test_case_fn = i
            
            is_ok = check_test_case(problem_id, test_case_fn, student_code_problem, student_code_parse,
                assignment_id=assignment_id, timeout_limit=timeout_limit,
                verbose=verbose, student_folder=student_folder, is_resubmit=is_resubmit,
            )
            if is_ok:  # Passed!
                passed_visible_cases.append(i)
            else:  # Failed!
                failed_visible_cases.append(i)

        visible_score = 100. * len(passed_visible_cases) / len(passed_visible_cases + failed_visible_cases)
        return {'visible_score': visible_score}


# Customized for different assignments
def check_test_case(problem_id, test_case_id, student_code_problem, student_code_parse, 
        assignment_id: str = 'a1',
        verbose: bool = False,
        timeout_limit: float = 5 * 60 + 1,
        student_folder: str = "",
        is_resubmit: bool = False,
        case_cond: dict = None,
        ) -> bool:
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join(assignment_id, 'test_cases','p'+str(problem_id))
    
    sup_printer.student_folder = student_folder

    try:
        error = None
        parse_timeout = TimeoutHelper(
            logger, student_code_parse, os.path.join(path,file_name_problem,))
        parse_time_limit = 3
        parse_timeout.join(parse_time_limit)
        if parse_timeout.is_alive():
            parse_timeout.stop()
            del parse_timeout
            raise TimeoutError(f"Parsing time exceeding {parse_time_limit} seconds!")
        problem = parse_timeout.result[0]
        if parse_timeout.error is not None:
            other_error = parse_timeout.error
            logger.error(f"{student_folder} (resubmit={is_resubmit}): Error at: {parse_timeout.line_info}")
            raise other_error[0](other_error[1])
    except Exception as e:  # handle error here
        error = e
    finally:
        if error is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): Found Parse ERROR: ({error}). '
                        f'Giving zero score for problem={problem_id}, '
                        f'test_case={test_case_id}.')
            return False
    
    if case_cond is None:
        win_rate = 0  # not used
        param = ()
        num_trials = 1
    else:
        win_rate = case_cond['win_rate']
        param: tuple = case_cond['param'][:-1]
        num_trials = case_cond['param'][-1]  # the last one indicates num_trials

    try:
        error = None
        timeout_helper = TimeoutHelper(
            logger, student_code_problem, problem, param, 
            num_trials=num_trials)
        timeout_helper.join(timeout_limit)
        if timeout_helper.is_alive():
            timeout_helper.stop()
            # timeout_helper.join()
            del timeout_helper  # avoid memory leak
            raise TimeoutError(f"Running time exceeding {timeout_limit} seconds!")
        student_solution = timeout_helper.result

        if timeout_helper.error is not None:
            other_error = timeout_helper.error
            logger.error(f"{student_folder} (resubmit={is_resubmit}): Error at: {timeout_helper.line_info}")
            raise other_error[0](other_error[1])

        # P1,P2,P3
        if problem_id in (1, 2, 3):
            student_solution = student_solution[0]  # since wrapped with list
            return cmp_solutions(path, file_name_sol, student_solution, test_case_id)
        # P2,P4,P5,P6
        else:
            raise NotImplementedError("Problem ID not supported")
    except Exception as e:  # handle error here
        error = e
    finally:
        if error is not None:
            msg = f'{student_folder} (resubmit={is_resubmit}): Found Run ERROR: ({error}). ' \
                f'Giving zero score for problem={problem_id}, ' \
                f'test_case={test_case_id}.'
            if isinstance(error, TimeoutError) and args.debug:
                logger.debug(msg)
            else:
                logger.warning(msg)
            return False
        return True


def cmp_solutions(path, file_name_sol, student_solution, test_case_id, verbose=False):
    solution = ''
    with open(os.path.join(path,file_name_sol)) as file_sol:
        solution = file_sol.read()
    if solution == student_solution:
        if (verbose):
            print('---------->', 'Test case', test_case_id, 'PASSED', '<----------')
        return True
    else:
        if (verbose):
            print('---------->', 'Test case', test_case_id, 'FAILED', '<----------')
            print('Your solution')
            print(student_solution)
            print('Correct solution')
            print(solution)
            for i,s in enumerate(difflib.ndiff(student_solution, solution)):
                if s[0]==' ': continue
                elif s[0]=='-':
                    print(u'Delete "{}" from position {}'.format(s[-1],i))
                elif s[0]=='+':
                    print(u'Add "{}" to position {}'.format(s[-1],i))
        return False


# Customized for different assignments
def a3_runtime_test(root, student_path, student_score_dict, max_time, eval_dir='a3/eval_env',
        is_resubmit: bool = False, is_supress_all: bool = True
        ):
    student_folder = os.path.basename(student_path)

    # sup_all_printer = SuprressPrint('', False)  # only used when is resubmitted
    sup_printer = SuprressPrint(student_folder, True, logger=logger)  # default: True; for debug, set False

    sup_printer.close()
    # sup_all_printer.close()

    # Copy students' files into eval_env
    eval_dir = os.path.join(root, eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    problem_fns = [f'{x}.py' for x in ['parse', 'p1', 'p2', 'p3', 'p4',]]
    customized_proj_fns = [s for s in os.listdir(student_path) if '.py' in s and 'grader' not in s]
    customized_proj_fns = list(filter(lambda x: x not in problem_fns, customized_proj_fns))
    if len(customized_proj_fns) > 0:
        logger.warning(f'[Warning] {student_folder} found other .py files: {customized_proj_fns}, need to check.')

    try:
        for p_fn in problem_fns + customized_proj_fns:
            shutil.copy(os.path.join(student_path, f'{p_fn}'), eval_dir)
    except Exception as e:
        logger.error(f'{student_folder}: {e}, giving zero for the assignment 1')

    sys.path.append(eval_dir)

    try:
        import parse
        importlib.reload(parse)
    except Exception as e:
        logger.error(f'{student_folder}: {e}, giving zero for the assignment 1')
    
    cur_scores: List = []

    try:
        import p1
        importlib.reload(p1)
        grade_out = grade(1, -8, p1.play_episode, parse.read_grid_mdp_problem_p1, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit,
            )
        s1 = grade_out['visible_score']
        s2 = 0.
        e = None
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 1')
    cur_scores.append((s1,))

    try:
        e = None
        import p2
        importlib.reload(p2)
        grade_out = grade(2, -7, p2.policy_evaluation, parse.read_grid_mdp_problem_p2, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit,
            )
        s1 = grade_out['visible_score']
        s2 = 0.
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 2')
    cur_scores.append((s1,))

    try:
        e = None
        import p3
        importlib.reload(p3)
        grade_out = grade(3, -4, p3.value_iteration, parse.read_grid_mdp_problem_p3, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit,
            )
        s1 = grade_out['visible_score']
        s2 = 0.
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 3')
    cur_scores.append((s1,))

    # give problem 4 for zero scores by default
    cur_scores.append((0,))

    assignment_score = calc_assignment_score(cur_scores, 
                                             max_problem_score_sums=[100,100,100,100])
    cur_scores.append(assignment_score)
    # logger.info(student_folder, cur_scores)

    if is_resubmit:
        # 1. Resubmission, use Dict to store scores
        student_score_dict[student_folder] = {
            'resubmit': cur_scores,  # [(s1,s2),...,avg], resubmitted
        }
    else:
        # 2. First submission
        cached_resubmit_score = None
        if student_score_dict.get(student_folder) is not None:
            cached_resubmit_score = student_score_dict[student_folder].get('resubmit')
        student_score_dict[student_folder] = {
            'first': cur_scores,  # [(s1,s2),...,avg]
            'resubmit': cached_resubmit_score,  # unchanged or None
        }

    sup_printer.open()
    # sup_all_printer.open()

    shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    return


# Customized for different assignments
def read_manual_score(fn='a3/manual.csv', logger=None):
    import csv
    try:
        with open(fn, 'r', newline='', encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter=',')
            csv_content = [row for row in f_csv]
        manual_dict = {}
        for row_idx, row in enumerate(csv_content):
            if row_idx == 0: continue  # headers
            student_folder, first_name, last_name, student_number, p4_reason, p4_score, final_score = row
            if p4_score != '':
                manual_dict[student_number] = (float(p4_score), str(p4_reason), str(final_score))
    except Exception as e:
        logger.warning(f'[Warning] {fn} not found, using default manual score. {e}')
        manual_dict = {}
    return manual_dict


# Customized for different assignments
def read_saved_score(fn='a3_scores.csv', logger=None):
    import csv
    try:
        with open(fn, 'r', newline='', encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter=',')
            csv_content = [row for row in f_csv]
        saved_dict = {}
        headers = []
        for r_idx, row in enumerate(csv_content):
            if r_idx == 0:  # headers
                saved_dict['headers'] = headers = row
            else:
                student_number = str(row[3])
                saved_dict[student_number] = {headers[i]: row[i] for i in range(len(row))}
    except Exception as e:
        logger.warning(f'[Warning] {fn} not found, using default score dict.')
        headers = ['student_id', 'first_name', 'last_name', 'student_number', 'group', 'email',
            'p1_vis',
            'p2_vis',
            'p3_vis',
            'p4_manual',
            'auto_score',
            'auto_discounted',
            'final_reason',
            'final_score',
            'feedback',
        ]
        saved_dict = {
            'headers': headers,  # students are empty
        }
    return saved_dict


# Customized for different assignments
def read_scores_from_csv(assignment: str):
    import csv
    email_dict = read_emails(logger=logger)

    # default values
    score_dict = {}
    for full_name in email_dict.keys():
        row_dict = {}
        row_dict["student_id"] = None
        row_dict["first_name"] = email_dict[full_name]['first_name']
        row_dict["last_name"] = email_dict[full_name]['last_name']
        row_dict["student_number"] = email_dict[full_name]['student_number']
        row_dict["group"] = email_dict[full_name]['group']
        row_dict["email"] = email_dict[full_name]['email']
        score_dict[row_dict['student_number']] = row_dict

    # read existing values to cover default values
    saved_score_dict = read_saved_score(f'{assignment}_scores.csv', logger=logger)
    for student_number in score_dict.keys():
        if student_number in saved_score_dict.keys():
            score_dict[student_number].update(saved_score_dict[student_number])

    score_dict['headers'] = saved_score_dict['headers']
    return score_dict


# Customized for different assignments
def save_scores_as_csv(assignment, student_score_dict, all_score_dict):
    """
    assignment: str
    student_score_dict: {`student_folder`:{'first':List, 'resubmit':List}}
    all_score_dict: {'headers':List, `student_number`:Dict}
    """
    import csv
    email_dict = read_emails(logger=logger)
    discount_dict = read_discount(f'{assignment}/discount.csv', logger=logger)  # set discounting ratio for different students
    manual_dict = read_manual_score(logger=logger)

    submit_dict = {k : False for k in email_dict.keys()}
    save_fn = f'{assignment}_scores.csv'
    headers = all_score_dict['headers']

    # update student_score_dict into all_score_dict
    for student_folder, scores in student_score_dict.items():
        student_full_name = student_folder.split('_')[0]
        if student_full_name in submit_dict.keys():
            submit_dict[student_full_name] = True
        if email_dict.get(student_full_name) is None:
            email_dict[student_full_name] = {
                'first_name': 'None',
                'last_name': 'None',
                'student_number': str(len(email_dict) + 1),
                'group': 'None',
                'email': 'None',
            }
        student_number = email_dict[student_full_name]['student_number']
        student_id = student_folder

        if student_number not in all_score_dict:
            all_score_dict[student_number] = {}

        all_score_dict[student_number].update({
            'student_id': str(student_folder),
            'first_name': str(email_dict[student_full_name]['first_name']),
            'last_name': str(email_dict[student_full_name]['last_name']),
            'student_number': str(email_dict[student_full_name]['student_number']),
            'group': str(email_dict[student_full_name]['group']),
            'email': str(email_dict[student_full_name]['email']),
        })

        first_score: List = scores['first']
        resubmit_score: List = scores['resubmit']
        key_of_score: List = headers[6:]
        idx_of_score = 0
        if resubmit_score is None:
            # without resubmission
            for score in first_score:
                if isinstance(score, tuple):
                    for part_score in score:
                        # visible score or tier ratio
                        all_score_dict[student_number].update({
                            key_of_score[idx_of_score] : part_score
                        })
                        idx_of_score += 1
                else:
                    # averaged score of assignment
                    all_score_dict[student_number].update({
                        key_of_score[idx_of_score] : score
                    })
                    idx_of_score += 1
            before_resubmit_avg, auto_avg = '', first_score[-1]
            discount_reason = ''
        else:
            # resubmission detected (TODO: Not implemented)
            first_avg, resubmit_avg = first_score[-1], resubmit_score[-1]
            higher_score = resubmit_score if first_avg < resubmit_avg else first_score
            discount_ratio, discount_reason = discount_dict.get(student_id, (0.9, 'resubmit'))
            auto_avg = max(first_avg, first_avg + (resubmit_avg - first_avg) * discount_ratio)
            for score in higher_score:  # higher one
                if isinstance(score, tuple):
                    one_row.append(f'{score[0]:.2f}')  # visible score
                    one_row.append(f'{score[1]:.2f}')  # invisible score
                else:
                    one_row.append(f'{score:.2f}')  # averaged score of assignment
            before_resubmit_avg, auto_avg = f'{first_score[-1]:.2f}', auto_avg
        
        # check plagiarism info
        if discount_dict.get(student_id) is not None:
            discount_ratio, discount_reason = discount_dict[student_id]
            if discount_reason == 'plagiarism':
                auto_avg *= 0.  # give zero
        # auto_score
        all_score_dict[student_number].update({
            'auto_score': auto_avg
        })

        # concat all info into a string
        row_dict =  all_score_dict[student_number]
        sum_info = [f'{x}:{row_dict[x]}' for x in row_dict.keys() if x in headers[6:-2]]
        sum_info = ', '.join(reversed(sum_info))
        all_score_dict[student_number].update({
            'feedback': sum_info
        })

    # read manually marked score
    for student_number in manual_dict.keys():
        p4_manual_score, p4_manual_reason, final_manual_score = manual_dict.get(student_number, (0, 'not given yet', ''))
        if 'feedback' in all_score_dict[student_number].keys():
            # final_reason, final_score
            final_reason_left_idx = all_score_dict[student_number]['feedback'].find("final_reason:")
            final_reason_right_idx = all_score_dict[student_number]['feedback'].find(" auto_score:")  # begins with a space
            final_reason_to_be_replaced = all_score_dict[student_number]['feedback'][final_reason_left_idx: final_reason_right_idx]
            # auto_score
            auto_score_left_idx = all_score_dict[student_number]['feedback'].find("auto_score:")
            auto_score_right_idx = all_score_dict[student_number]['feedback'].find(" p4_manual:")  # begins with a space
            auto_score_to_be_replaced = all_score_dict[student_number]['feedback'][auto_score_left_idx: auto_score_right_idx]
            # p4_manual
            p4_manual_left_idx = all_score_dict[student_number]['feedback'].find("p4_manual:")
            p4_manual_right_idx = all_score_dict[student_number]['feedback'].find(" p3_vis:")  # begins with a space
            p4_manual_to_be_replaced = all_score_dict[student_number]['feedback'][p4_manual_left_idx: p4_manual_right_idx]
            # udpate scores
            p1_score = float(all_score_dict[student_number]['p1_vis'] if all_score_dict[student_number]['p1_vis'] != '' else 0)
            p2_score = float(all_score_dict[student_number]['p2_vis'] if all_score_dict[student_number]['p2_vis'] != '' else 0)
            p3_score = float(all_score_dict[student_number]['p3_vis'] if all_score_dict[student_number]['p3_vis'] != '' else 0)
            auto_score = (p1_score + p2_score + p3_score) / 3
            final_score = (p1_score + p2_score + p3_score + float(p4_manual_score)) / 4
            final_reason = f"`p4 {p4_manual_reason}`"
            if final_reason_to_be_replaced != '':
                feedback = all_score_dict[student_number]['feedback'].replace(
                        final_reason_to_be_replaced, f'final_reason:{final_reason},') if \
                            final_reason_to_be_replaced != '' else all_score_dict[student_number]['feedback']
            else:
                feedback = f"final_reason:{final_reason}, {all_score_dict[student_number]['feedback']}"
            feedback = feedback.replace(
                    auto_score_to_be_replaced, f'auto_score:{auto_score:.2f},') if auto_score_to_be_replaced != '' else feedback
            feedback = feedback.replace(
                    p4_manual_to_be_replaced, f'p4_manual:{p4_manual_score},') if p4_manual_to_be_replaced != '' else feedback
            if feedback == '': feedback = 'not submitted'
            all_score_dict[student_number].update({
                'p4_manual': p4_manual_score,
                'auto_score': auto_score,
                'final_reason': final_reason,
                'final_score': final_score,
                'feedback': feedback,
            })

    # save all_score_dict into .csv file
    csv_rows = []

    no_headers_score_dict = {k: v for k, v in all_score_dict.items() if k != 'headers'}
    items = list(no_headers_score_dict.items())
    items.sort(key=lambda x: x[1]['last_name'] + x[1]['first_name'])
    no_headers_score_dict = dict(items)

    for student_number in no_headers_score_dict.keys():
        if student_number == 'headers': continue
        row_dict = all_score_dict[student_number]
        one_row = []
        for col_key in headers:
            item_val = row_dict.get(col_key, '')
            if isinstance(item_val, (str, int)):
                one_row.append(item_val)
            elif isinstance(item_val, float):
                one_row.append(f'{item_val:.2f}')
            else:
                one_row.append('')
        csv_rows.append(one_row)

    with open(save_fn, 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(csv_rows)


if __name__ == "__main__":
    args = argparse.ArgumentParser("Auto grading for Assignment 3.")
    args.add_argument('-r', '--root', type=str, default='D:/Documents/HKU/COMP7404_Marking/',
        help='Where you place your "a3/submissions/" and "a3/eval_env/", ')
    args.add_argument('-a', '--assignment', type=str, default='a3')
    args.add_argument('--load_resubmit', action='store_true',
        help='Whether check resubmitted files.')
    args.add_argument('--debug', action='store_true')
    args.add_argument('--skip', action="store_true")
    args.add_argument('--chose', action="store_true")
    args.add_argument('--manual', action="store_true")
    args = args.parse_args()
    logger = init_logger(f"{args.assignment}_output.log")
    sup_printer.logger = logger

    root = args.root
    assignment = args.assignment
    if args.debug:
        max_time = 0.1  # 61 for debug, 301 for real marking
    else:
        max_time = 1.1
    chose_students = []
    if args.chose:
        chose_students = [
            "AtYuan Ge_assign",
        ]
    skip_students = ['error_student3', 'example_student1', 'example_student2', 'AtYuan Ge_assign']
    submission_dir = make_abs_path(os.path.join(root, assignment, 'submissions/'))
    resubmission_dir = make_abs_path(os.path.join(root, assignment, 'resubmit_after_ddl/'))

    # 0. Reading existing scores.csv or creating new empty score dict (if .csv not exists)
    all_score_dict = read_scores_from_csv(assignment=assignment)  # key:student_number, value:row_dict

    # 1. Calculating scores case-by-case
    submission_fns = os.listdir(submission_dir)
    submission_fns = [x for x in submission_fns if '.DS' not in x]  # remove .DS_Store on MAC
    # submission_fns = [x for x in submission_fns if 'Yuan Ge' not in x]  # remove TA's data
    if args.chose and len(chose_students) > 0:
        skip_students = [x for x in submission_fns if x not in chose_students]
    submission_fns.sort()
    skip = args.skip  # True for debug
    for idx, student_folder in enumerate(submission_fns):
        if student_folder == 'AtYuan Ge_assign':
            skip = False
        if skip: continue
        if student_folder in skip_students:
            continue

        student_score_dict = {}

        if not args.manual:
            # Start auto-grading
            # a. Second submission, read resubmitted files first to avoid re-printing errors
            logger.info('===' * 10 + " " + student_folder + f" ({idx}/{len(submission_fns)}) " + '===' * 10)
            has_resubmitted = False
            if args.load_resubmit:
                student_path = os.path.join(resubmission_dir, student_folder)
                if os.path.exists(student_path):
                    has_resubmitted = True
                    a3_runtime_test(root, student_path, student_score_dict, max_time,
                        is_resubmit=True)
            # b. First submission
            student_path = os.path.join(submission_dir, student_folder)
            if os.path.isdir(student_path):
                a3_runtime_test(root, student_path, student_score_dict, max_time,
                    is_supress_all=has_resubmitted)

        # c. After getting score for a student, save results into a .csv file at once
        save_scores_as_csv(
            assignment=assignment,
            student_score_dict=student_score_dict,
            all_score_dict=all_score_dict,
        )

        if args.manual: # No need to continue iterating
            exit()

