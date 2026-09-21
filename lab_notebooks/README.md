# Publishing labs to DataHub

Students open labs from the course schedule using **Open Lab 01 in DataHub**. Do not publish starter ZIP/download links.

## Lab 01

The released files live together in `lab01/`: `Lab01.ipynb`, `club_event_survey.csv`, `requirements.txt`, and `tests/lab01/q*.py`. The source notebook was copied from the separate authoring folder without changing its reviewed content.

The launcher URL is configured at `links.labs.lab01` in `_config.yml` and displayed in `_modules/week-02 2.md`. It uses nbgitpuller to pull the public `dssdecal/fa26` repository's `main` branch into the student's general Berkeley DataHub account, then opens `fa26/lab_notebooks/lab01/Lab01.ipynb` in JupyterLab. The CSV and tests arrive automatically. Students run the notebook from top to bottom and use its Otter checks. Submission remains the bCourses process described in the notebook.

DataHub's published base environment currently specifies Otter 6.1.6 and pandas 3.0.0. The lab requirements accept those versions; nbgitpuller does not install requirements. If a particular server cannot import pandas or Otter, staff should check its kernel/environment before asking students to install packages.

## Release checks

1. Keep the notebook, CSV, and tests in the same relative layout. Clear student answer cells and execution outputs before release. Keep instructor solutions outside this repository: nbgitpuller pulls the entire repository.
2. Run a completed copy outside the public repository. Confirm each `grader.check(...)` and the final `grader.check_all()` pass.
3. Commit and push the lab files and schedule link together. GitHub Actions builds and publishes the website on pushes to `main`.
4. Follow the link from the published Week 2 schedule in a signed-in DataHub session. Run setup, verify CSV loading, and check an exercise before announcing release.
5. Once students have started, avoid unnecessary restructuring or renaming of notebook cells/files.

## How Data 8 and Data 100 compare

Both courses use website links to nbgitpuller, public student-material repositories, DataHub notebooks, and Otter checks. Data 8 Fall 2026 uses `data-8/materials-fa26`; Data 100 Fall 2026 uses `DS-100/fa26-student`. Their launch URLs target course-specific hubs; this DeCal targets the general hub. They keep website and assignment repositories separate. For this first DeCal lab, we use the existing site repository; the student click-to-open workflow is the same. A separate materials repository can be introduced later.

Data 8's Lab 01 also exports a completed submission ZIP with Otter for its grading platform. That is a submission artifact, not a starter bundle students download to begin. This DeCal retains its existing bCourses notebook submission instructions.

Sources:
- https://data8.org/fa26/
- https://github.com/data-8/materials-fa26/tree/main/lab/lab01
- https://ds100.org/fa26/
- https://github.com/DS-100/fa26-student
- https://github.com/berkeley-dsep-infra/base-python-image/blob/main/environment.yml
- https://curriculum-guide.datahub.berkeley.edu/workflows/distributing-files/
