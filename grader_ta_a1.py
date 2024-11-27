#Do not make changes to this file
import os, difflib, sys, copy
import time, argparse, threading, importlib, multiprocessing, shutil
import logging
import functools, inspect
import builtins, traceback
from typing import List, Dict


def init_logger(log_path: str):
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_path, mode="a+", encoding="utf-8")
    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    console_handler.setLevel("DEBUG")
    file_handler.setFormatter(formatter)
    file_handler.setLevel("INFO")
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel("INFO")
    return logger


class SuprressPrint(object):  # shut down printing
    def __init__(self, student_folder, enable=True, logger=None):
        self.student_folder = student_folder
        self.enable = enable
        self.called_time = 0
        self.logger = logger

        self.original_print = None
        self.original_open = None

        self.opened = True

    def close(self):
        if not self.enable: return
        if not self.opened: return
        self.close_print()
        self.lock_functions()
        self.opened = False

    def open(self):
        if not self.enable: return
        if self.opened: return
        self.unlock_functions()
        self.open_print()
        self.opened = True

    def close_print(self):  # shut down print
        self.called_time += 1
        self.original_print = builtins.print
        def _block_print(*args, **kwargs):
            pass
        builtins.print = _block_print

    def open_print(self):
        self.called_time += 1
        builtins.print = self.original_print

    def lock_functions(self):
        self.original_open = builtins.open
        def _block_open(*args, **kwargs):
            mode = None
            banned_actions = ("w", "w+", "a", "a+")
    
            # Check kwargs for 'mode'
            if 'mode' in kwargs and kwargs['mode'] in banned_actions:
                mode = kwargs['mode']
            
            # Check the second parameter of args (if present)
            if len(args) > 1 and args[1] in banned_actions:
                mode = args[1]

            # Check args for a dictionary containing 'mode'
            for arg in args:
                if isinstance(arg, dict) and 'mode' in arg and arg['mode'] in banned_actions:
                    mode = arg['mode']
                    break

            if mode is not None:
                raise PermissionError(f"Attempt to call `open(..., mode='{mode})'`.")
                
            return self.original_open(*args, **kwargs)
        builtins.open = _block_open

    def unlock_functions(self):
        builtins.open = self.original_open


class TimeoutHelper(threading.Thread):  # time-limit helper
    def __init__(self, logger, fun, problem: any, args: tuple = (), 
                 num_trials: int = 1,
                 student_folder: str = "",
                 ):
        super(TimeoutHelper, self).__init__()
        self.setDaemon(True)
        self.logger = logger
        self.result = []
        self.error = None
        self.line_info = None
        self.fun = fun
        self.problem = problem
        self.args = args
        self.num_trials = num_trials
        self.student_folder = student_folder
        self.stop_event = threading.Event()  # Create a stop event

        assert isinstance(args, tuple), f'Input args should be wrapped with Tuple, but found: {type(args)} '
        self.start()

    def run(self):
        for i in range(self.num_trials):
            in_problem = copy.deepcopy(self.problem)
            in_args = copy.deepcopy(self.args)

            # Get the function signature
            signature = inspect.signature(self.fun)
            default_count = 0
            for param in signature.parameters.values():  # how many params having default values
                if param.default != param.empty:
                    default_count += 1
            num_params = len(signature.parameters) - default_count

            if isinstance(in_problem, tuple) and num_params != 1 + len(in_args):
                in_func = functools.partial(self.fun, *in_problem)  # Unpack arguments
            else:
                in_func = functools.partial(self.fun, in_problem)  # No need to unpack

            if len(in_args) > 0:  # Append other auguments
                in_func = functools.partial(in_func, *in_args)
            
            try:
                result = in_func()
            except:
                self.error = sys.exc_info()
                exc_type, exc_value, exc_traceback = sys.exc_info()
                for filename, lineno, name, line in traceback.extract_tb(exc_traceback):
                    self.line_info = f"file={os.path.basename(filename)}, line={lineno}: {line}"
                in_problem = None
                del in_problem
            finally:
                if self.error is not None:
                    self.result = [None] * self.num_trials
                    break
                else:
                    self.result.append(result)

    def stop(self):
        self.stop_event.set()  # Method to set the stop event


sup_printer = SuprressPrint("")


# Customized for different assignments
def grade(problem_id, test_case_id, student_code_problem, student_code_parse,
        assignment_id: str = 'a1',
        verbose: bool = False,
        num_invisible_cases: int = 3,
        timeout_limit: float = 5 * 60 + 1,  # max running time for each testing case is 5 min
        student_folder: str = "",
        is_resubmit: bool = False,
        ):
    if verbose:
        print('Grading Problem', problem_id,':')
    if test_case_id > 0:
        #single test case
        check_test_case(problem_id, test_case_id, student_code_problem, student_code_parse)
    else:
        #multiple test cases
        num_test_cases = test_case_id * (-1)
        passed_visible_cases, failed_visible_cases = [], []  # visible testing cases
        passed_invisible_cases, failed_invisible_cases = [], []  # invisible testing cases
        # former testing cases are visible
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
        # the last 3 testing cases are invisible
        invis_idx = 101  # invisble ids start from 101
        for i in range(0, num_invisible_cases):
            test_case_fn = i + invis_idx
            is_ok = check_test_case(problem_id, test_case_fn, student_code_problem, student_code_parse,
                assignment_id=assignment_id, timeout_limit=timeout_limit,
                verbose=verbose, student_folder=student_folder, is_resubmit=is_resubmit,
            )
            if is_ok:  # Passed!
                passed_invisible_cases.append(i)
            else:  # Failed!
                failed_invisible_cases.append(i)
        # print(passed_visible_cases, failed_visible_cases)
        # print(passed_invisible_cases, failed_invisible_cases)
        visible_score = 50. * len(passed_visible_cases) / len(passed_visible_cases + failed_visible_cases)
        invisible_score = 50. * len(passed_invisible_cases) / len(passed_invisible_cases + failed_invisible_cases)
        return visible_score, invisible_score


# Customized for different assignments
def check_test_case(problem_id, test_case_id, student_code_problem, student_code_parse, 
        assignment_id: str = 'a1',
        verbose: bool = False,
        timeout_limit: float = 5 * 60 + 1,
        student_folder: str = "",
        is_resubmit: bool = False,
        ) -> bool:
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join(assignment_id, 'test_cases','p'+str(problem_id))

    sup_printer.student_folder = student_folder

    try:
        error = None
        parse_timeout = TimeoutHelper(
            logger, student_code_parse, (os.path.join(path,file_name_problem,)))
        parse_timeout.join(11)
        if parse_timeout.is_alive():
            raise TimeoutError("Parsing time exceeding 11 seconds!")
        problem = parse_timeout.result[0]
        if parse_timeout.error is not None:
            other_error = parse_timeout.error
            raise other_error[0](other_error[1])
    except Exception as e:  # handle error here
        error = e
    finally:
        if error is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): Found ERROR: ({error}). '
                f'Giving zero score for problem={problem_id}, '
                f'test_case={test_case_id}.')
            return False

    try:
        error = None
        timeout_helper = TimeoutHelper(
            logger, student_code_problem, problem)
        timeout_helper.join(timeout_limit)
        if timeout_helper.is_alive():
            raise TimeoutError(f"Running time exceeding {timeout_limit} seconds!")
        student_solution = timeout_helper.result[0]
        if timeout_helper.error is not None:
            other_error = timeout_helper.error
            raise other_error[0](other_error[1])
    except Exception as e:  # handle error here
        error = e
    finally:
        if error is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): Found ERROR: ({error}). '
                f'Giving zero score for problem={problem_id}, '
                f'test_case={test_case_id}.')
            return False

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
def a1_runtime_test(root, student_path, student_score_dict, max_time, eval_dir='a1/eval_env',
        is_resubmit: bool = False, has_resubmitted: bool = False
        ):
    student_folder = os.path.basename(student_path)

    # sup_all_printer = SuprressPrint('', has_resubmitted, logger=logger)
    sup_printer = SuprressPrint(student_folder, not args.chose, logger=logger)

    # sup_all_printer.close()
    sup_printer.close()

    # Copy students' files into eval_env
    eval_dir = os.path.join(root, eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    problem_fns = [f'{x}.py' for x in ['parse', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7',]]
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
        s1, s2 = grade(1, -5, p1.dfs_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit)
        e = None
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 1')
    cur_scores.append((s1, s2))

    try:
        e = None
        import p2
        importlib.reload(p2)
        s1, s2 = grade(2, -5, p2.bfs_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 2')
    cur_scores.append((s1, s2))

    try:
        e = None
        import p3
        importlib.reload(p3)
        s1, s2 = grade(3, -6, p3.ucs_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 3')
    cur_scores.append((s1, s2))

    try:
        e = None
        import p4
        importlib.reload(p4)
        s1, s2 = grade(4, -6, p4.greedy_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 4')
    cur_scores.append((s1, s2))

    try:
        e = None
        import p5
        importlib.reload(p5)
        s1, s2 = grade(5, -6, p5.astar_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 5')
    cur_scores.append((s1, s2))

    try:
        e = None
        import p6
        importlib.reload(p6)
        s1, s2 = grade(6, -4, p6.number_of_attacks, parse.read_8queens_search_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 6')
    cur_scores.append((s1, s2))

    try:
        e = None
        import p7
        importlib.reload(p7)
        s1, s2 = grade(7, -6, p7.better_board, parse.read_8queens_search_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 7')
    cur_scores.append((s1, s2))

    assignment_score = calc_assignment_score(cur_scores)
    cur_scores.append(assignment_score)
    # print(student_folder, cur_scores)

    if is_resubmit:
        # 1. Resubmission, use Dict to store scores
        student_score_dict[student_folder] = {
            'resubmit': cur_scores,  # [(s1,s2),...,s_avg], resubmitted
        }
    else:
        # 2. First submission
        cached_resubmit_score = None
        if student_score_dict.get(student_folder) is not None:
            cached_resubmit_score = student_score_dict[student_folder].get('resubmit')
        student_score_dict[student_folder] = {
            'first': cur_scores,  # [(s1,s2),...,s_avg]
            'resubmit': cached_resubmit_score,  # unchanged or None
        }

    sup_printer.open()
    # sup_all_printer.open()

    shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)

    logger.info(f"Finished evaluation (is_resubmit={is_resubmit})")
    print(cur_scores)

    return


def make_abs_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


def read_emails(fn='emails.csv', logger=None):
    import csv
    try:
        with open(fn, 'r', newline='', encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter=',')
            csv_content = [row for row in f_csv]
        email_dict = {}
        for row in csv_content[1:]:  # skip header
            first_name, last_name, student_number, _, _, email_addr, _, _ = row
            email_dict[f'{last_name} {first_name}'] = {
                'first_name': first_name,
                'last_name': last_name,
                'student_number': student_number,
                'group': 'None',
                'email': email_addr,
            }
    except Exception as e:
        logger.warning(f'[Warning] emails.csv not found, using empty meta info. Error={e}')
        email_dict = {}
    return email_dict


def read_discount(fn='a1/discount.csv', logger=None):
    import csv
    try:
        with open(fn, 'r', newline='', encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter=',')
            csv_content = [row for row in f_csv]
        discount_dict = {}
        for row in csv_content:
            student_folder, ratio, reason = row
            discount_dict[student_folder] = (float(ratio), str(reason))
    except Exception as e:
        logger.warning('[Warning] discount.csv not found, using default discounting ratio.')
        discount_dict = {}
    return discount_dict


def calc_assignment_score(problem_scores: list, max_problem_score_sums: list = None):
    num_problems = len(problem_scores)
    sum_score = 0.
    for i in range(len(problem_scores)):
        # Sum should be 100
        one_problem_score = problem_scores[i]
        max_problem_score_sum = 100 if max_problem_score_sums is None else max_problem_score_sums[i]
        problem_score = 0.
        for score in one_problem_score:
            problem_score += score
        sum_score += problem_score * (100 / max_problem_score_sum)
    return sum_score / num_problems


# Customized for different assignments
def save_scores_as_csv(assignment, student_score_dict):
    import csv
    email_dict = read_emails(logger=logger)
    submit_dict = {k : False for k in email_dict.keys()}
    save_fn = f'{assignment}_scores.csv'
    headers = ['student_id', 'first_name', 'last_name', 'student_number', 'group', 'email',
               'p1_vis', 'p1_invis',
               'p2_vis', 'p2_invis',
               'p3_vis', 'p3_invis',
               'p4_vis', 'p4_invis',
               'p5_vis', 'p5_invis',
               'p6_vis', 'p6_invis',
               'p7_vis', 'p7_invis',
               'latest_score',
               'before_resubmit',
               'discount_reason',
               'final_score',
               'sum',
               ]
    csv_rows = []
    discount_dict = read_discount(logger=logger) # set discounting ratio for different students
    for student_id, scores in student_score_dict.items():
        student_full_name = student_id.split('_')[0]
        if student_full_name in submit_dict.keys():
            submit_dict[student_full_name] = True
        if email_dict.get(student_full_name) is None:
            email_dict[student_full_name] = {
                'first_name': 'None',
                'last_name': 'None',
                'student_number': 'None',
                'group': 'None',
                'email': 'None',
            }
        one_row = [str(student_id), 
            str(email_dict[student_full_name]['first_name']),
            str(email_dict[student_full_name]['last_name']),
            str(email_dict[student_full_name]['student_number']),
            str(email_dict[student_full_name]['group']),
            str(email_dict[student_full_name]['email']),
        ]
        first_score: List = scores['first']
        resubmit_score: List = scores['resubmit']
        if resubmit_score is None:
            # without resubmission
            for score in first_score:
                if isinstance(score, tuple):
                    one_row.append(f'{score[0]:.2f}')  # visible score
                    one_row.append(f'{score[1]:.2f}')  # invisible score
                else:
                    one_row.append(f'{score:.2f}')  # averaged score of assignment
            before_resubmit_avg, final_avg = '', first_score[-1]
            discount_reason = ''
        else:
            # resubmission detected
            first_avg, resubmit_avg = first_score[-1], resubmit_score[-1]
            higher_score = resubmit_score if first_avg < resubmit_avg else first_score
            discount_ratio, discount_reason = discount_dict.get(student_id, (0.9, 'resubmit'))
            final_avg = max(first_avg, first_avg + (resubmit_avg - first_avg) * discount_ratio)
            for score in higher_score:  # higher one
                if isinstance(score, tuple):
                    one_row.append(f'{score[0]:.2f}')  # visible score
                    one_row.append(f'{score[1]:.2f}')  # invisible score
                else:
                    one_row.append(f'{score:.2f}')  # averaged score of assignment
            before_resubmit_avg, final_avg = f'{first_score[-1]:.2f}', final_avg
        
        # check plagiarism info
        if discount_dict.get(student_id) is not None:
            discount_ratio, discount_reason = discount_dict[student_id]
            if 'plagiarism' in discount_reason:
                final_avg *= 0.  # give zero
        one_row.extend([before_resubmit_avg, discount_reason, f'{final_avg:.2f}'])

        # concat all info into a string
        sum_info = [f'{x}:{y}' for x, y in zip(headers[6:-2], one_row[6:-1])]
        sum_info = ', '.join(reversed(sum_info))
        one_row.append(sum_info)

        csv_rows.append(one_row)

    # append unsubmitted students
    for student_full_name, is_submitted in submit_dict.items():
        if not is_submitted:
            logger.warning(f'[Warning] {student_full_name} has not submitted the code.')
            one_row = [
                str(f'{student_full_name}_no_submission'), 
                str(email_dict[student_full_name]['first_name']),
                str(email_dict[student_full_name]['last_name']),
                str(email_dict[student_full_name]['student_number']),
                str(email_dict[student_full_name]['group']),
                str(email_dict[student_full_name]['email']),
            ] + ['0.00' for _ in range(7 * 2)] + ['0.00', '', 'not submitted', '0.00', 'not submitted']
            csv_rows.append(one_row)

    with open(save_fn, 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(csv_rows)


if __name__ == "__main__":
    args = argparse.ArgumentParser("Auto grading for Assignment 1.")
    args.add_argument('-r', '--root', type=str, default='D:/Documents/HKU/COMP7404_Marking/',
        help='Where you place your "a1/submissions/" and "a1/eval_env/", ')
    args.add_argument('-a', '--assignment', type=str, default='a1')
    args.add_argument('--load_resubmit', action='store_true',
        help='Whether check resubmitted files.')
    args.add_argument('--debug', action='store_true')
    args.add_argument('--skip', action="store_true")
    args.add_argument('--chose', action="store_true")
    args = args.parse_args()
    logger = init_logger(f"{args.assignment}_output.log")
    sup_printer.logger = logger

    root = args.root
    assignment = args.assignment
    if args.debug:
        max_time = 1.1  # 1 for debug, 301 for real marking
    else:
        max_time = 301
    chose_students = []
    if args.chose:
        chose_students = [
            "Han Mengchi_17992097_assignsubmission_file",
        ]
    skip_students = ['error_student3', 'example_student1', 'example_student2']
    submission_dir = make_abs_path(os.path.join(root, assignment, 'submissions/'))
    resubmission_dir = make_abs_path(os.path.join(root, assignment, 'resubmit_after_ddl/'))

    # 1. Calculating scores case-by-case
    student_score_dict = {}
    submission_fns = os.listdir(submission_dir)
    submission_fns = [x for x in submission_fns if 'assign' in x]
    submission_fns = [x for x in submission_fns if 'Yuan Ge' not in x]  # remove TA's data
    if args.chose and len(chose_students) > 0:
        skip_students = [x for x in submission_fns if x not in chose_students]
    submission_fns.sort()
    skip = args.skip  # True for debug
    for idx, student_folder in enumerate(submission_fns):
        if student_folder == 'Yang Jiahao_18025337_assignsubmission_file':
            skip = False
        if skip: continue
        if student_folder in skip_students:
            continue
        logger.info('===' * 10 + " " + student_folder + f" ({idx}/{len(submission_fns)}) " + '===' * 10)
        # a. Second submission, read resubmitted files first to avoid re-printing errors
        has_resubmitted = False
        if args.load_resubmit:
            student_path = os.path.join(resubmission_dir, student_folder)
            if os.path.exists(student_path):
                has_resubmitted = True
                a1_runtime_test(root, student_path, student_score_dict, max_time,
                    is_resubmit=True)
        # b. First submission
        student_path = os.path.join(submission_dir, student_folder)
        if os.path.isdir(student_path):
            a1_runtime_test(root, student_path, student_score_dict, max_time,
                has_resubmitted=has_resubmitted)

    # 2. Save results into a .csv file
    if not args.chose:
        save_scores_as_csv(
            assignment=assignment,
            student_score_dict=student_score_dict,
        )
    else:
        pass
