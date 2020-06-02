# How to Contribute

Anyone who wishes to contribute to this repository is welcome! For contributing new code to this repository, we roughly follow a [gitflow workflow](https://nvie.com/posts/a-successful-git-branching-model). We also support [forking workflows](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow) for contributors who wish to fork this repository and maintain their own local versions. 

So, when contributing, please make your changes in a new feature branch that is branched off of 'dev', and submit this branch as a [pull request](https://github.com/int-brain-lab/sorting_analysis/pulls). Please lint your code (see below) before submitting a pull request.

# Style Guidelines

We follow pep8 conventions, and we lint using [flake8](https://flake8.pycqa.org/en/latest/), with specifications in our [flake8 config file](https://github.com/int-brain-lab/sorting_analysis/blob/master/.flake8). To use flake8 to lint the code in this repository, simply run `flake8 <path/to/sorting_analysis/>` where you replace `<path/to/sorting_analysis>` with the full path in which you've cloned this repository.