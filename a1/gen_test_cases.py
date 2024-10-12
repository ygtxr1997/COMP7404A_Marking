import os, parse, difflib
import p1, p2, p3, p4, p5, p6, p7


def gen_sol(problem_id, test_case_ids, student_code_problem, student_code_parse):
    for test_case_id in test_case_ids:
        if test_case_id > 0:
            if test_case_id < 100:
                print(f'[Problem={problem_id}] Skipping test case: {test_case_id}')
                continue  # Do not change visible testing cases
            # single test case
            gen_test_case(problem_id, test_case_id, student_code_problem, student_code_parse)

def gen_test_case(problem_id, test_case_id, student_code_problem, student_code_parse):
    file_name_problem = str(test_case_id)+'.prob' 
    file_name_sol = str(test_case_id)+'.sol'
    path = os.path.join('test_cases','p'+str(problem_id)) 
    problem = student_code_parse(os.path.join(path,file_name_problem))
    student_solution = student_code_problem(problem)
    with open(os.path.join(path,file_name_sol), 'w') as file_sol:
        file_sol.write(student_solution)


if __name__ == "__main__":
    gen_cases = [101, 102, 103]
    gen_sol(1, gen_cases, p1.dfs_search, parse.read_graph_search_problem)
    gen_sol(2, gen_cases, p2.bfs_search, parse.read_graph_search_problem)
    gen_sol(3, gen_cases, p3.ucs_search, parse.read_graph_search_problem)
    gen_sol(4, gen_cases, p4.greedy_search, parse.read_graph_search_problem)
    gen_sol(5, gen_cases, p5.astar_search, parse.read_graph_search_problem)
    gen_sol(6, gen_cases, p6.number_of_attacks, parse.read_8queens_search_problem)
    gen_sol(7, gen_cases, p7.better_board, parse.read_8queens_search_problem)
