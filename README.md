## Introduction

This repository includes the scripts of marking scores of the assignments of COMP7404A.


## Usage for Students

If you'd like to test your code using our auto grading script, you can follow these steps (assuming your chosen directory is *YOUR_ROOT/*):

1. If you haven't cloned this repo, run `git clone`:
```shell
cd YOUR_ROOT/
git clone https://github.com/ygtxr1997/COMP7404A_Marking.git
```
If you have already cloned this repo before, run `git pull` or `git update`:
```shell
cd YOUR_ROOT/COMP7404A_Marking/
git pull  # or `git update`
```

2. Enter the dir and copy your code to submissions directory.
Assuming your `p*.py` files are placed at *YOUR_OWN_NAME/* directory.
Then you can:
```shell
cd COMP7404A_Marking/
cd a1/  # or others assignment
mkdir submissions  # create a folder to include your all files
cp YOUR_OWN_NAME/ submissions/  # e.g tom_jerry_12241235
```

3. Add your own testing cases:

Since we have some **invisible testing cases** that are invisible for you, you need to add these cases by yourself. Each problem has 3 invisible cases with the same name patterns:
```shell
101.prob, 101.sol, 102.prob, 102.sol, 103.prob, 103.sol
```
For each problem, you should put these `10*.prob` and `10*.sol` files into `a1/test_cases/p*` directory.

After the above steps, the directory tree should be like this:
```shell
COMP7404A_Marking/  # under YOUR_ROOT/ folder
    |---a1/
        |---submissions/
            |---YOUR_OWN_NAME/  # any customized name is OK
                |---p*.py
                |---parse.py
        |---test_cases/
            |---p*/
                |---*.prob
                |---*.sol
                |---10*.prob  # invisible testing cases definied by yourself
                |---10*.sol  # solutions defined by yourself
```

4. Start evaluation:
```shell
python grader_ta_a1.py -r YOUR_ROOT -a a1
```
Then a csv file named `a1_scores.csv` will be saved at *YOUR_ROOT/*.


## Usage for TA

`grader.py`: an empty file, just for the importing action of students' p*.py

`grader_ta_a1.py`: the marking script of the assignment a1

`preprocess_submissions.py`: unzip submitted files downloaded from Moodle


## Directory Tree

The directory tree of this repo is shown below:

```shell
# This is our GitHub repo
COMP7404A_Marking/
    |---a*/  # assignment id
        |---test_cases/  # testing cases
            |---p*/  # each problem has visible and invisible cases
                |---*.prob
                |---*.sol
            ...
    ...
    |---grader.py  # just for avoiding importing error
    |---grader_ta_a1.py  # marking for assignment 1

# This is where you place your code
YOUR_OWN_ROOT/
    |---a*/  # assignment id
        |---submissions/  # students' submissions are placed here
                |---student_id1/  # each student is a folder
                    |---p*.py
                    |---parse.py
                |---student_id2/
                    |---p*.py
                    |---parse.py
                ...
        |---eval_env/  # auto grading scripts will run codes read from here
```

## Update Log

**2024/10/12:** update `grader_ta_a1.py` to handle more unexpected errors; 
add `preprocess_submissions.py` to unzip submitted files; add `a1/gen_test_cases.py` to generate solutions for invisible testing cases.

## TODO

- [X] Add auto-unzip script

- [ ] Add auto-grader for a2

## Note

If you have any questions about this repo, please open an issue here,
or make a post on our ED forum, or drop me an email.
