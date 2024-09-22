## Introduction

This repository includes the scripts of marking scores of the assignments of COMP7404A.

## Update Log



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
```

## Usage

`grader.py`: an empty file, just for the importing action of students' p*.py

`grader_ta_a1.py`: the marking script of the assignment a1

## TODO

- [ ] Add auto-unzip script

- [ ] Add auto-grader for a2
