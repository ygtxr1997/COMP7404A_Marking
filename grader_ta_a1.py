#Do not make changes to this file
import os, difflib, sys
import time, argparse, threading, importlib, multiprocessing, shutil
from typing import List, Dict


class SuprressPrint(object):  # shut down printing
    def __init__(self, student_folder, enable=True):
        self.student_folder = student_folder
        self.enable = enable
    def __enter__(self):
        self.close()
    def __exit__(self, exc_type, exc_value, traceback):
        self.open()
    def close(self):
        if not self.enable: return
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def open(self):
        if not self.enable: return
        sys.stdout.close()
        sys.stdout = self._original_stdout


class TimeoutHelper(threading.Thread):  # time-limit helper
    def __init__(self, fun, args):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.result = None
        self.error = None
        self.fun = fun
        self.args = args
 
        self.start()

    def run(self):
        try:
            self.result = self.fun(self.args)
        except:
            self.error = sys.exc_info()


def grade(problem_id, test_case_id, student_code_problem, student_code_parse,
        assignment_id: str = 'a1',
        verbose: bool = False,
        num_invisible_cases: int = 3,
        timeout_limit: float = 5 * 60 + 1,  # max running time for each testing case is 5 min
        sup_printer: SuprressPrint = None,
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
                verbose=verbose, sup_printer=sup_printer, is_resubmit=is_resubmit,
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
                verbose=verbose, sup_printer=sup_printer, is_resubmit=is_resubmit,
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


def check_test_case(problem_id, test_case_id, student_code_problem, student_code_parse, 
        assignment_id: str = 'a1',
        verbose: bool = False,
        timeout_limit: float = 5 * 60 + 1,
        sup_printer: SuprressPrint = None,
        is_resubmit: bool = False,
        ) -> bool:
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join(assignment_id, 'test_cases','p'+str(problem_id))

    try:
        parse_timeout = TimeoutHelper(student_code_parse, os.path.join(path,file_name_problem))
        parse_timeout.join(61)
        if parse_timeout.is_alive():
            raise TimeoutError("Parsing time exceeding 1 min!")
        problem = parse_timeout.result
        if parse_timeout.error is not None:
            other_error = parse_timeout.error
            raise other_error[0](other_error[1])
    except Exception as e:  # handle error here
        sup_printer.open()
        print(f'{sup_printer.student_folder} (resubmit={is_resubmit}): Found ERROR: ({e}). '
              f'Giving zero score for problem={problem_id}, '
              f'test_case={test_case_id}.')
        sup_printer.close()
        return False

    try:
        timeout_helper = TimeoutHelper(student_code_problem, problem)
        timeout_helper.join(timeout_limit)
        if timeout_helper.is_alive():
            raise TimeoutError("Running time exceeding 5 min!")
        student_solution = timeout_helper.result
        if timeout_helper.error is not None:
            other_error = timeout_helper.error
            raise other_error[0](other_error[1])
    except Exception as e:  # handle error here
        sup_printer.open()
        print(f'{sup_printer.student_folder} (resubmit={is_resubmit}): Found ERROR: ({e}). '
              f'Giving zero score for problem={problem_id}, '
              f'test_case={test_case_id}.')
        sup_printer.close()
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


def a1_runtime_test(root, student_path, student_score_dict, max_time, eval_dir='a1/eval_env',
        is_resubmit: bool = False, is_supress_all: bool = False
        ):
    student_folder = os.path.basename(student_path)

    sup_all_printer = SuprressPrint('', is_supress_all)
    sup_printer = SuprressPrint(student_folder, True)

    sup_all_printer.close()

    # Copy students' files into eval_env
    eval_dir = os.path.join(root, eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    problem_fns = [f'{x}.py' for x in ['parse', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7',]]
    customized_proj_fns = [s for s in os.listdir(student_path) if '.py' in s and 'grader' not in s]
    customized_proj_fns = list(filter(lambda x: x not in problem_fns, customized_proj_fns))
    if len(customized_proj_fns) > 0:
        print(f'[Warning] {student_folder} found other .py files: {customized_proj_fns}, need to check.')

    try:
        for p_fn in problem_fns + customized_proj_fns:
            shutil.copy(os.path.join(student_path, f'{p_fn}'), eval_dir)
    except Exception as e:
        print(f'{student_folder}: {e}, giving zero for the assignment 1')

    sys.path.append(eval_dir)

    try:
        import parse
        importlib.reload(parse)
    except Exception as e:
        print(f'{student_folder}: {e}, giving zero for the assignment 1')
    
    cur_scores: List = []

    try:
        sup_printer.close()
        import p1
        importlib.reload(p1)
        s1, s2 = grade(1, -5, p1.dfs_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, sup_printer=sup_printer, is_resubmit=is_resubmit)
        e = None
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        sup_printer.open()
        if e is not None:
            print(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 1')
    cur_scores.append((s1, s2))

    try:
        sup_printer.close()
        import p2
        importlib.reload(p2)
        s1, s2 = grade(2, -5, p2.bfs_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, sup_printer=sup_printer, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        sup_printer.open()
        if e is not None:
            print(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 2')
    cur_scores.append((s1, s2))

    try:
        sup_printer.close()
        import p3
        importlib.reload(p3)
        s1, s2 = grade(3, -6, p3.ucs_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, sup_printer=sup_printer, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        sup_printer.open()
        if e is not None:
            print(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 3')
    cur_scores.append((s1, s2))

    try:
        sup_printer.close()
        import p4
        importlib.reload(p4)
        s1, s2 = grade(4, -6, p4.greedy_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, sup_printer=sup_printer, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        sup_printer.open()
        if e is not None:
            print(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 4')
    cur_scores.append((s1, s2))

    try:
        sup_printer.close()
        import p5
        importlib.reload(p5)
        s1, s2 = grade(5, -6, p5.astar_search, parse.read_graph_search_problem, 
            timeout_limit=max_time, sup_printer=sup_printer, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        sup_printer.open()
        if e is not None:
            print(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 5')
    cur_scores.append((s1, s2))

    try:
        sup_printer.close()
        import p6
        importlib.reload(p6)
        s1, s2 = grade(6, -4, p6.number_of_attacks, parse.read_8queens_search_problem, 
            timeout_limit=max_time, sup_printer=sup_printer, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        sup_printer.open()
        if e is not None:
            print(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 6')
    cur_scores.append((s1, s2))

    try:
        sup_printer.close()
        import p7
        importlib.reload(p7)
        s1, s2 = grade(7, -6, p7.better_board, parse.read_8queens_search_problem, 
            timeout_limit=max_time, sup_printer=sup_printer, is_resubmit=is_resubmit)
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        sup_printer.open()
        if e is not None:
            print(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 7')
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

    sup_all_printer.open()

    shutil.rmtree(eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    return


def make_abs_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


def read_emails(fn='emails.csv'):
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
        print(f'[Warning] emails.csv not found, using empty meta info. Error={e}')
        email_dict = {}
    return email_dict


def read_discount(fn='a1_discount.csv'):
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
        print('[Warning] discount.csv not found, using default discounting ratio.')
        discount_dict = {}
    return discount_dict


def calc_assignment_score(problem_scores: list):
    num_problems = len(problem_scores)
    sum_score = 0.
    for score_vis, score_invis in problem_scores:
        sum_score += score_vis + score_invis
    return sum_score / num_problems


def save_scores_as_csv(assignment, student_score_dict):
    import csv
    email_dict = read_emails()
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
    discount_dict = read_discount() # set discounting ratio for different students
    for student_id, scores in student_score_dict.items():
        student_full_name = student_id.split('_')[0]
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
            if discount_reason == 'plagiarism':
                final_avg *= 0.  # give zero
        one_row.extend([before_resubmit_avg, discount_reason, f'{final_avg:.2f}'])

        # concat all info into a string
        sum_info = [f'{x}:{y}' for x, y in zip(headers[6:-2], one_row[6:-1])]
        sum_info = ', '.join(reversed(sum_info))
        one_row.append(sum_info)

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
    args = args.parse_args()
    root = args.root
    assignment = args.assignment
    if args.debug:
        max_time = 1  # 1 for debug, 301 for real marking
    else:
        max_time = 301
    skip_students = ['error_student3', 'example_student1', 'example_student2']
    submission_dir = make_abs_path(os.path.join(root, assignment, 'submissions/'))
    resubmission_dir = make_abs_path(os.path.join(root, assignment, 'resubmit_after_ddl/'))

    # 1. Calculating scores case-by-case
    student_score_dict = {}
    submission_fns = os.listdir(submission_dir)
    submission_fns = [x for x in submission_fns if 'assign' in x]
    submission_fns.sort()
    for idx, student_folder in enumerate(submission_fns):
        if student_folder in skip_students:
            continue
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
                is_supress_all=has_resubmitted)

    # 2. Save results into a .csv file
    save_scores_as_csv(
        assignment=assignment,
        student_score_dict=student_score_dict,
    )
