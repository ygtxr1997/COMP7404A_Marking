#Do not make changes to this file
import os, difflib, sys
import time, argparse, threading, importlib
# import parse, p1, p2, p3, p4, p5, p6, p7

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
        ):
    if verbose:
        print('Grading Problem',problem_id,':')
    if test_case_id > 0:
        #single test case
        check_test_case(problem_id, test_case_id, student_code_problem, student_code_parse)
    else:
        #multiple test cases
        num_test_cases = test_case_id * (-1)
        passed_visible_cases, failed_visible_cases = [], []  # visible testing cases
        passed_invisible_cases, failed_invisible_cases = [], []  # invisible testing cases
        for i in range(1, num_test_cases+1):
            is_ok = check_test_case(problem_id, i, student_code_problem, student_code_parse,
                assignment_id=assignment_id, timeout_limit=timeout_limit,
                verbose=verbose)
            if i <= num_test_cases - num_invisible_cases:  # former testing cases are visible
                passed_cases, failed_cases = passed_visible_cases, failed_visible_cases
            else:  # the last 3 testing cases are invisible
                passed_cases, failed_cases = passed_invisible_cases, failed_invisible_cases
            if is_ok:  # Passed!
                passed_cases.append(i)
            else:  # Failed!
                failed_cases.append(i)
        # print(passed_visible_cases, failed_visible_cases)
        # print(passed_invisible_cases, failed_invisible_cases)
        visible_score = 50. * len(passed_visible_cases) / len(passed_visible_cases + failed_visible_cases)
        invisible_score = 50. * len(passed_invisible_cases) / len(passed_invisible_cases + failed_invisible_cases)
        return visible_score, invisible_score

def check_test_case(problem_id, test_case_id, student_code_problem, student_code_parse, 
        assignment_id: str = 'a1',
        verbose: bool = False,
        timeout_limit: float = 5 * 60 + 1,
        ) -> bool:
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join(assignment_id, 'test_cases','p'+str(problem_id)) 
    problem = student_code_parse(os.path.join(path,file_name_problem))

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
        print(f'Found ERROR: ({e}). Giving zero score for problem={problem_id}, '
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
                
def make_abs_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

def calc_assignment_score(problem_scores: list):
    num_problems = len(problem_scores)
    sum_score = 0.
    for score_vis, score_invis in problem_scores:
        sum_score += score_vis + score_invis
    return sum_score / num_problems

if __name__ == "__main__":
    assignment = 'a1'
    max_time = 301  # 1 for debug, 301 for real marking
    submission_dir = make_abs_path(os.path.join(assignment, 'submissions/'))

    # 1. Calculating scores case-by-case
    student_score_dict = {}
    for student_folder in os.listdir(submission_dir):
        student_path = os.path.join(submission_dir, student_folder)
        if os.path.isdir(student_path):
            sys.path.append(student_path)
            import parse, p1, p2, p3, p4, p5, p6, p7
            importlib.reload(parse)
            importlib.reload(p1)
            importlib.reload(p2)
            importlib.reload(p3)
            importlib.reload(p4)
            importlib.reload(p5)
            importlib.reload(p6)
            importlib.reload(p7)
            
            student_score_dict[student_folder] = []

            s1, s2 = grade(1, -5, p1.dfs_search, parse.read_graph_search_problem, 
                timeout_limit=max_time)
            student_score_dict[student_folder].append((s1, s2))

            s1, s2 = grade(2, -5, p2.bfs_search, parse.read_graph_search_problem, 
                timeout_limit=max_time)
            student_score_dict[student_folder].append((s1, s2))

            s1, s2 = grade(3, -6, p3.ucs_search, parse.read_graph_search_problem, 
                timeout_limit=max_time)
            student_score_dict[student_folder].append((s1, s2))

            s1, s2 = grade(4, -6, p4.greedy_search, parse.read_graph_search_problem, 
                timeout_limit=max_time)
            student_score_dict[student_folder].append((s1, s2))

            s1, s2 = grade(5, -6, p5.astar_search, parse.read_graph_search_problem, 
                timeout_limit=max_time)
            student_score_dict[student_folder].append((s1, s2))

            s1, s2 = grade(6, -4, p6.number_of_attacks, parse.read_8queens_search_problem, 
                timeout_limit=max_time)
            student_score_dict[student_folder].append((s1, s2))

            s1, s2 = grade(7, -6, p7.better_board, parse.read_8queens_search_problem, 
                timeout_limit=max_time)
            student_score_dict[student_folder].append((s1, s2))

            assignment_score = calc_assignment_score(student_score_dict[student_folder])
            student_score_dict[student_folder].append(assignment_score)
            print(student_folder, student_score_dict[student_folder])

            sys.path.remove(student_path)

    # 2. Save results into a .csv file
    import csv
    save_fn = f'{assignment}_scores.csv'
    headers = ['student_id', 
               'p1_vis', 'p1_invis',
               'p2_vis', 'p2_invis',
               'p3_vis', 'p3_invis',
               'p4_vis', 'p4_invis',
               'p5_vis', 'p5_invis',
               'p6_vis', 'p6_invis',
               'p7_vis', 'p7_invis',
               'assignment'
               ]
    csv_rows = []
    for student_id, scores in student_score_dict.items():
        one_row = [str(student_id)]
        for score in scores:
            if isinstance(score, tuple):
                one_row.append(f'{score[0]:.2f}')  # visible score
                one_row.append(f'{score[1]:.2f}')  # invisible score
            else:
                one_row.append(f'{score:.2f}')  # averaged score of assignment
        csv_rows.append(one_row)
    with open(save_fn, 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(csv_rows)
