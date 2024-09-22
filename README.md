## Introduction

This repository includes the scripts of marking scores of the assignments of COMP7404A.


## Usage

`grader.py`: an empty file, just for the importing action of students' p*.py

`grader_ta_a1.py`: the marking script of the assignment a1


## Directory Tree

The directory tree of this repo is shown below:

```shell
COMP7404A_Marking/
    |---a*/  # assignment id
        |---submissions/  # students' submissions are placed here
            |---student_id1/  # each student is a folder
                |---p*.py
                |---parse.py
            |---student_id2/
                |---p*.py
                |---parse.py
            ...
        |---test_cases/  # testing cases
            |---p*/  # each problem has visible and invisible cases
                |---*.prob
                |---*.sol
            ...
    ...
    |---grader.py  # just for avoiding importing error
    |---grader_ta_a1.py  # marking for assignment 1
```

## Update Log


## TODO

- [ ] Add auto-unzip script

- [ ] Add auto-grader for a2

## Note

If you have any questions about this repo, please open an issue here,
or make a post on our ED forum, or drop me an email.
