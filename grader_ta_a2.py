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


class TierHelper(object):
    def __init__(self):
        self.tiers = [None, 'tier1', 'tier2', 'tier3']
        self.a2_tier_dict = {
            2: {
                # 'tier':[(case_id, (arguments,), win_rate)]
                'tier3': [(1, (10,), 0)],
                'tier2': [(2, (10,), 30), 
                        (3, (10,), 30), 
                        (4, (10,), 30)],
                'tier1': [(5, (10,), 40)]
            },
            4: {
                # 'tier':[(case_id, (arguments,), win_rate)]
                'tier3': [(1, (10,), 0)],
                'tier2': [(4, (100,), 5),
                        (7, (100,), 5),
                        (5, (100,), 5),],
                'tier1': [(9, (10,), 5)]
            },
            5: {
                # 'tier':[(case_id, (arguments,), win_rate)]
                'tier3': [(1, (2, 1), 0)],
                'tier2': [(2, (7, 1), 0),
                        (7, (3, 1), 0),],
                'tier1': [(4, (2, 1), 0), (8, (4, 1), 0)]
            },
            6: {
                # 'tier':[(case_id, (arguments,), win_rate)]
                'tier3': [(1, (1, 1), 0)],
                'tier2': [(6, (2, 1), 0), ],
                'tier1': [(3, (5, 8), 0), (5, (3, 1), 0)]
            }
        }  # problem: 'tier': [(case, (trials) or (depth,trials), win_rate)]

    def get_tier_conditon(self, problem_id: int, tier_id: int, num_cases: int):
        assert tier_id in range(1, len(self.tiers))
        tier_cond: List = self.a2_tier_dict[problem_id][self.tiers[tier_id]]

        case_ids = []
        params = []
        win_rates = []
        for case_cond in tier_cond:
            case, param, win_rate = case_cond
            if case == 'all':
                case_ids = list(range(1, num_cases + 1))
                params = [param for _ in range(num_cases)]
                win_rates = [win_rate for _ in range(num_cases)]
                break
            case_ids.append(case)
            params.append(param)
            win_rates.append(win_rate)

        return {
            'case_id': case_ids,
            'param': params,
            'win_rate': win_rates,
        }

    def is_tier_mode_problem(self, problem_id: int):
        return problem_id in self.a2_tier_dict.keys()



# Customized for different assignments
def grade(problem_id, test_case_id, student_code_problem, student_code_parse,
        assignment_id: str = 'a2',
        verbose: bool = False,
        timeout_limit: float = 5 * 60 + 1,  # max running time for each testing case is 5 min
        student_folder: str = "",
        is_resubmit: bool = False,
        tier_helper: TierHelper = None,
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
        is_tier_mode = tier_helper.is_tier_mode_problem(problem_id)

        # 1. Vanilla testing mode
        if not is_tier_mode:
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
        
        # 2. Tier testing mode
        else:
            tiers_passed_cases = []
            tiers_failed_cases = []

            iter_order = list(range(1, 3 + 1))
            for tier_id in iter_order:  # tier3 > tier2 > tier1
                cases_cond = tier_helper.get_tier_conditon(problem_id, tier_id, num_test_cases)
                case_ids = cases_cond['case_id']
                params = cases_cond['param']  # (trials) or (depth,trials)
                win_rates = cases_cond['win_rate']
                tier_passed_cases, tier_failed_cases = [], []

                for i, case_id in enumerate(case_ids):
                    test_case_fn = case_id
                    is_ok = check_test_case(problem_id, test_case_fn, student_code_problem, student_code_parse,
                        assignment_id=assignment_id, timeout_limit=timeout_limit,
                        verbose=verbose, student_folder=student_folder, is_resubmit=is_resubmit,
                        case_cond={'param': params[i],
                                   'win_rate': win_rates[i]},
                    )

                    # # Try different meaning of depth (ply or summed)
                    # if len(params[i]) >= 2:  # (depth,trials)
                    #     ply_depth = (1 + 4) * params[0]
                    #     ply_param = (ply_depth, *params[:-1])
                    #     ply_ok = check_test_case(problem_id, test_case_fn, student_code_problem, student_code_parse,
                    #         assignment_id=assignment_id, timeout_limit=timeout_limit,
                    #         verbose=verbose, sup_printer=sup_printer, is_resubmit=is_resubmit,
                    #         case_cond={'param': ply_param,
                    #                 'win_rate': win_rates[i]},
                    #     )
                    #     is_ok = is_ok or ply_ok

                    if is_ok:
                        tier_passed_cases.append(case_id)
                    else:
                        tier_failed_cases.append(case_id)

                tiers_passed_cases.append(tier_passed_cases)
                tiers_failed_cases.append(tier_failed_cases)

                # If tier1 passed, tier 2 and 3 will not be checked
                if len(tier_failed_cases) == 0:
                    break
            
            # Append 100 as padding
            for i in range(len(tiers_passed_cases), len(iter_order)):
                tiers_passed_cases.append([100] * 3)
                tiers_failed_cases.append([])

            tiers_passed_ratio = [
                len(tiers_passed_cases[i]) / (len(tiers_passed_cases[i]) + len(tiers_failed_cases[i])) * 100 
                for i in range(len(tiers_passed_cases))
            ]  # 
            return {f'tier{tier_id + 1}_passed': tiers_passed_ratio[tier_id] 
                    for tier_id in range(len(tiers_passed_ratio))}


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

        # P1,P3
        if problem_id in (1, 3):
            student_solution = student_solution[0]  # since wrapped with list
            return cmp_solutions(path, file_name_sol, student_solution, test_case_id)
        # P2,P4,P5,P6
        else:
            assert problem_id in (2, 4, 5, 6), "Problem ID not supported"
            win_count = 0
            for trial_solution in student_solution:  # num_trials*[(out1,out2,...)]
                assert len(trial_solution) == 2, "Student's output #params > 2"
                _, winner = trial_solution
                if winner == 'Pacman': win_count += 1
            win_p = win_count / len(student_solution) * 100
            # logger.info(f"Problem={problem_id}, Case={test_case_id}, Win_Rate={win_rate}")
            return win_p >= win_rate * 0.95  # loose the win rate condition
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
                logger.error(msg)
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
def a2_runtime_test(root, student_path, student_score_dict, max_time, eval_dir='a2/eval_env',
        is_resubmit: bool = False, is_supress_all: bool = True
        ):
    student_folder = os.path.basename(student_path)

    # sup_all_printer = SuprressPrint('', False)  # only used when is resubmitted
    sup_printer = SuprressPrint(student_folder, True, logger=logger)  # default: True; for debug, set False
    tier_helper = TierHelper()

    sup_printer.close()
    # sup_all_printer.close()

    # Copy students' files into eval_env
    eval_dir = os.path.join(root, eval_dir)
    os.makedirs(eval_dir, exist_ok=True)
    problem_fns = [f'{x}.py' for x in ['parse', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6',]]
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
        # sup_printer.close()
        import p1
        importlib.reload(p1)
        grade_out = grade(1, -6, p1.random_play_single_ghost, parse.read_layout_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit,
            tier_helper=tier_helper)
        s1 = grade_out['visible_score']
        s2 = 0.
        e = None
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        # sup_printer.open()
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 1')
    cur_scores.append((s1,))

    try:
        e = None
        import p2
        importlib.reload(p2)
        grade_out = grade(2, -5, p2.better_play_single_ghosts, parse.read_layout_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit,
            tier_helper=tier_helper)
        s1, s2, s3 = grade_out['tier3_passed'], grade_out['tier2_passed'], grade_out['tier1_passed']
    except Exception as error:
        e = error
        s1, s2, s3 = 0., 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 2')
    cur_scores.append((s3, s2, s1))

    try:
        e = None
        import p3
        importlib.reload(p3)
        grade_out = grade(3, -7, p3.random_play_multiple_ghosts, parse.read_layout_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit,
            tier_helper=tier_helper)
        s1 = grade_out['visible_score']
        s2 = 0.
    except Exception as error:
        e = error
        s1, s2 = 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 3')
    cur_scores.append((s1,))

    try:
        e = None
        import p4
        importlib.reload(p4)
        grade_out = grade(4, -9, p4.better_play_multiple_ghosts, parse.read_layout_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit,
            tier_helper=tier_helper)
        s1, s2, s3 = grade_out['tier3_passed'], grade_out['tier2_passed'], grade_out['tier1_passed']
    except Exception as error:
        e = error
        s1, s2, s3 = 0., 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 4')
    cur_scores.append((s3, s2, s1))

    try:
        e = None
        import p5
        importlib.reload(p5)
        grade_out = grade(5, -9, p5.min_max_multiple_ghosts, parse.read_layout_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit,
            tier_helper=tier_helper)
        s1, s2, s3 = grade_out['tier3_passed'], grade_out['tier2_passed'], grade_out['tier1_passed']
    except Exception as error:
        e = error
        s1, s2, s3 = 0., 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 5')
    cur_scores.append((s3, s2, s1))

    try:
        e = None
        import p6
        importlib.reload(p6)
        grade_out = grade(6, -9, p6.expecti_max_multiple_ghosts, parse.read_layout_problem, 
            timeout_limit=max_time, student_folder=student_folder, is_resubmit=is_resubmit,
            tier_helper=tier_helper)
        s1, s2, s3 = grade_out['tier3_passed'], grade_out['tier2_passed'], grade_out['tier1_passed']
    except Exception as error:
        e = error
        s1, s2, s3 = 0., 0., 0.
    finally:
        if e is not None:
            logger.error(f'{student_folder} (resubmit={is_resubmit}): {e}, giving zero for the problem 6')
    cur_scores.append((s3, s2, s1))

    assignment_score = calc_assignment_score(cur_scores, 
                                             max_problem_score_sums=[100,300,100,300,300,300])
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


def read_manual_score(fn='a2/manual.csv'):
    import csv
    try:
        with open(fn, 'r', newline='', encoding='utf-8') as f:
            f_csv = csv.reader(f, delimiter=',')
            csv_content = [row for row in f_csv]
        manual_dict = {}
        for row in csv_content:
            student_folder, ratio, reason = row
            manual_dict[student_folder] = (float(ratio), str(reason))
    except Exception as e:
        logger.warning('[Warning] manual.csv not found, using default mannual score.')
        manual_dict = {}
    return manual_dict


# Customized for different assignments
def save_scores_as_csv(assignment, student_score_dict):
    import csv
    email_dict = read_emails()
    manual_dict = read_manual_score(f'{assignment}/manual.csv')
    submit_dict = {k : False for k in email_dict.keys()}
    save_fn = f'{assignment}_scores.csv'
    headers = ['student_id', 'first_name', 'last_name', 'student_number', 'group', 'email',
               'p1_vis',
               'p2_tier3', 'p2_tier2', 'p2_tier1',
               'p3_vis',
               'p4_tier3', 'p4_tier2', 'p4_tier1',
               'p5_tier3', 'p5_tier2', 'p5_tier1',
               'p6_tier3', 'p6_tier2', 'p6_tier1',
               'auto_score',
               'auto_discounted',
               'final_reason',
               'final_score',
               'feedback',
               ]
    csv_rows = []
    discount_dict = read_discount(f'{assignment}/discount.csv')  # set discounting ratio for different students
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
                    for part_score in score:
                        one_row.append(f'{part_score:.2f}')  # visible score or tier ratio
                else:
                    one_row.append(f'{score:.2f}')  # averaged score of assignment
            before_resubmit_avg, auto_avg = '', first_score[-1]
            discount_reason = ''
        else:
            # resubmission detected
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
        one_row.extend([f'{auto_avg:.2f}'])  # auto_score

        # read manually marked score
        final_manual_score, final_manual_reason = manual_dict.get(student_id, (0, 'not given'))
        one_row.extend([final_manual_reason, f'{final_manual_score:.2f}'])  # final_reason, final_score

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
            ] + ['0.00' for _ in range(2 * 1 + 4 * 3)] + ['0.00', 'not submitted', '0.00', 'not submitted']
            csv_rows.append(one_row)

    with open(save_fn, 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(csv_rows)


if __name__ == "__main__":
    args = argparse.ArgumentParser("Auto grading for Assignment 2.")
    args.add_argument('-r', '--root', type=str, default='D:/Documents/HKU/COMP7404_Marking/',
        help='Where you place your "a2/submissions/" and "a2/eval_env/", ')
    args.add_argument('-a', '--assignment', type=str, default='a2')
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
        max_time = 0.1  # 61 for debug, 301 for real marking
    else:
        max_time = 1.1
    chose_students = []
    if args.chose:
        chose_students = [
            "Cao Shuochen_18025314_assignsubmission_file",
        ]
    skip_students = ['error_student3', 'example_student1', 'example_student2']
    submission_dir = make_abs_path(os.path.join(root, assignment, 'submissions/'))
    resubmission_dir = make_abs_path(os.path.join(root, assignment, 'resubmit_after_ddl/'))

    # 1. Calculating scores case-by-case
    student_score_dict = {}
    submission_fns = os.listdir(submission_dir)
    submission_fns = [x for x in submission_fns if 'assign' in x]
    # submission_fns = [x for x in submission_fns if 'Yuan Ge' not in x]  # remove TA's data
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
                a2_runtime_test(root, student_path, student_score_dict, max_time,
                    is_resubmit=True)
        # b. First submission
        student_path = os.path.join(submission_dir, student_folder)
        if os.path.isdir(student_path):
            a2_runtime_test(root, student_path, student_score_dict, max_time,
                is_supress_all=has_resubmitted)

    # 2. Save results into a .csv file
    if not args.chose:
        save_scores_as_csv(
            assignment=assignment,
            student_score_dict=student_score_dict,
        )
    else:
        pass
