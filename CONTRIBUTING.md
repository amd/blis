## How to contribute to AOCL-BLAS

First, we want to thank you for your interest in contributing to AOCL-BLAS! Please read through the following guidelines to help you better understand how to best contribute your potential bug report, bugfix, feature, etc.

#### **Did you find a bug?**

* **Check if the bug has already been reported** by searching on GitHub under [Issues](https://github.com/amd/blis/issues).

* If you can't find an open issue addressing the problem, please feel free to [open a new one](https://github.com/amd/blis/issues/new). Some things to keep in mind as you create your issue:
   * Be sure to include a **meaningful title**. Aim for a title that is neither overly general nor overly specific.
   * Putting some time into writing a **clear description** will help us understand your bug and how you found it.
   * You are welcome to include the AOCL-BLAS version number (e.g. 0.3.2-15) if you wish, but please supplement it with the **actual git commit number** corresponding to the code that exhibits your reported behavior (the first seven or eight hex digits is fine).
   * Unless you are confident that it's not relevant, it's usually recommended that you **tell us how you configured AOCL-BLAS** and **about your environment in general**. Your hardware microarchitecture, OS, compiler (including version), `configure` options, configuration target are all good examples of things to you may wish to include. If the bug involves elements of the build system such as bash or python functionality, please include those versions numbers, too.
   * If your bug involves behavior observed after linking to AOCL-BLAS and running an application, please provide a minimally illustrative **code sample** that developers can run to (hopefully) reproduce the error or other concerning behavior.

#### **Did you write a patch that fixes a bug?**

If so, great, and thanks for your efforts! Please submit a new GitHub [pull request](https://github.com/amd/blis/pulls) with the patch.

* Ensure the PR description clearly describes the problem and solution. Include any relevant issue numbers, if applicable.

* Please limit your PR to addressing one issue at a time. For example, if you are fixing a bug and in the process you find a second, unrelated bug, please open a separate PR for the second bug (or, if the bugfix to the second bug is not obvious, you can simply open an [issue](https://github.com/amd/blis/issues/new) for the second bug).

* Before submitting new code, please read the [coding conventions](https://github.com/flame/blis/wiki/CodingConventions) guide to learn more about our preferred coding conventions. (It's unlikely that we will turn away your contributed code due to mismatched coding styles, but it will be **highly** appreciated by project maintainers since it will save them the time of digressing from their work--whether now or later--to reformat your code.)

#### **Did you fix whitespace or reformat code?**

Unlike some other projects, if you find code that does not abide by the project's [coding conventions](https://github.com/flame/blis/wiki/CodingConventions) and you would like to bring that code up to our standards, we will be happy to accept your contribution. Please note in the commit log the fixing of whitespace, formatting, etc. as applicable.

If you are making a more substantial contribution and in the vicinity of the affected code (i.e., within the same file) you stumble upon other code that works but could use some trivial changes or reformatting, you may combine the latter into the commit for the former. Just note in your commit log that you also fixed whitespace or applied reformatting.

#### **Do you intend to add a new feature or change an existing one?**

That's fine, we are interested to hear your ideas!

* You may wish to introduce your idea by opening an [issue](https://github.com/amd/blis/issues/new) to describe your new feature, or how an existing feature is not sufficiently general-purpose. This allows you the chance to open a dialogue with other developers, who may provide you with useful feedback.

* Before submitting new code, please read the [coding conventions](https://github.com/flame/blis/wiki/CodingConventions) guide to learn more about our preferred coding conventions. (See comments above regarding mismatched coding styles.)

#### How are external pull requests integrated?

This document outlines the process for handling external contributions to the AOCL-BLAS project, ensuring quality, transparency, and timely communication with contributors. The workflow covers PR validation, review, internal integration, and contributor notification.

**Process Overview:**

1. PR Submission:
   - External contributors submit a PR to the `dev` branch of the public repository: **https://github.com/amd/blis.git**.

2. Initial CI/CD Validation:
   - A CI/CD job runs on a Linux Docker container to check for Coverity (static analysis) issues.

3. Coverity Report Review:
   - The reviewer examines the Coverity report.
   - If issues are found, the PR is rejected, and details are communicated to the contributor via GitHub comments.
   - If there are no issues, the reviewer approves the Coverity check via the CI/CD URL received by email.

4. Presubmit CI/CD Jobs:
   - Upon approval, presubmit (sanity) tests are triggered.
   - Reviewers receive email notifications about the job status.

5. Presubmit Results Review:
   - If tests fail, the PR is rejected, and the contributor is notified via GitHub comments with relevant details.
   - If tests pass, the reviewer approves PR.

6. Internal Repository Integration:
   - The approved PR is ported to the AOCL-BLAS internal GitHub repository, preserving the contributor's name and email.

7. Internal Validation:
   - The PR undergoes regular internal CI/CD validations.
   - If issues are found, they are reported to the contributor via the external GitHub comment section.
   - If there are no issues, the PR is merged into the internal repository.

8. Release and Notification:
   - Merged changes are included in the subsequent open-source release (dev branch).
   - Contributors are notified via GitHub comments once their changes are merged.
   - Any issues found during internal release candidate (RC) or test cycles are handled by the internal team.

**Flow Diagram Description:**

```
[External Contributor]
   |
   v
[Submit PR to Public Repo (dev branch)]
   |
   v
[CI/CD: Coverity Scan]
   |
   v
[Review Coverity Report]
   /                  \
[Issues]            [No Issues]
   |                      |
   v                      v
[Reject PR]             [Approve Coverity]
   |                      |
   v                      |
[Notify Contributor]      |
                          v
              [CI/CD: Presubmit Tests]
                          |
                          v
              [Review Presubmit Results]
               /                   \
            [Fail]              [Pass]
             |                    |
             v                    v
       [Reject PR]          [Approve PR]
          |                        |
          v                        |
[Notify Contributor]               |
                                   v
                        [Port PR to Internal Repo]
                                   |
                                   v
                        [Internal CI/CD Validations]
                           /             \
                        [Fail]           [Pass]
                          |                |
                          v                v
                  [Notify Contributor]  [Merge PR]
                                           |
                                           v
                                 [Weekly Release to Public Repo]
                                           |
                                           v
                                 [Notify Contributor of Merge]
```

**Communication:**

- All rejection reasons and validation results are communicated to contributors via GitHub comments.
- Contributors are notified when their changes are merged and released.

**Notes:**

- Contributor attribution (name and email) is preserved during internal integration.
- Internal team is responsible for post-merge issue resolution during RC and test cycles.
- The internal team will act as reviewers and have the authority to either approve or reject pull requests (PRs).

Here at the AOCL-BLAS project, we :heart: our community. Thanks for helping to make AOCL-BLAS better:blush:!

— The AMD AOCL-BLAS Team
