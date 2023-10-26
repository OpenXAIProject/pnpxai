---
description: Why and How to Collaborate
---

# Collaboration Manual

## Summary

* Collaboration Tool
  * Git book : documentation tool
  * Git Project : Issue tracking tool

## Why we should use Collaboration Tools?

* Communication
  * Notification about your task to other team members
    * ex. what to do, when is the deadline
  * Ask and Answer question about other team members' task
    * ex. what is your output of your module?
  * Suggest Issues that should be discussed
    * ex. what is the best theme color for our product?
* Project Management
  * Monitoring
    * How much did we do? How much does it left to complete the product?
  * Reporting
    * What was the major milestone in last month

## Documentation : Gitbook

### Why we use this tool?

* Git Based Documentation Tool
* Good Version Control
* Both Web UI and Code based UI

## Project Management : Github Project

### Why we use this tool?

* Free to use
* Git based Issue tracking tool

## Workflow Guidelines

#### 1. Decision Making

* Begin by deciding on the task you need to undertake.

#### 2. Issue Creation

* Document your chosen task by creating an issue in the Git project.
* Provide a clear description of what you plan to accomplish.

#### 3. Referencing Documentation

* If additional information is required, refer to the documentation available on GitBook.
* In case of specific queries or if you need expert guidance, proceed to create a 'question issue' directed to the responsible person or team.

#### 4. Collaboration and Task Ownership

* Respect task ownership: Do not alter or complete tasks assigned to other team members without their consent.
* If collaboration or assistance is required, seek permission through a comment on the issue or direct communication.
* This is to avoid conflicts between codes.

#### 5. Working with Branches

* Create a new branch for your task, branching off from the main development branch.

#### 6. Creating Pull Requests

* Once your task is completed, create a pull request against the main development branch.
* Clearly describe the changes made and link to the corresponding issue for context.
* Assign or request reviewers for your pull request. A minimum of one or two reviewers, depending on the project's complexity, is recommended.

#### 7. Review and Merge Protocol

* Do not merge your pull requests. Wait for reviews and approval from assigned team members.
* Reviewers should provide constructive feedback and approve changes only when they meet the project’s standards and guidelines.
* If changes are requested, address these revisions in a timely manner and update the pull request.

#### 8. Completion and Updating Documentation

* Once your task is completed and the pull request is merged, update the relevant sections in GitBook.
  * You can do it by revising the documentation in gitbooks directory. It is not free to edit the Gitbook in Web UI for more than 2 members. Please notice that update of documentation in Gitbook should be on the `gitbooks` branch
* This ensures that changes and new information are easily accessible to others.

#### 9. Handling Critical or Discussable Issues

* Should you encounter any critical issues or topics that require team discussion, promptly raise them in the Git project.
* Make sure to share and highlight these issues in the Group Meeting for further deliberation and collective input.

#### 10. Continuous Communication

* Maintain open and continuous communication with the team.
* Regularly update the status of your tasks and any challenges you face in the Git project.

#### Note

* Open communication and regular updates in the Git project are crucial for smooth workflow and team synergy.
* The workflow is designed to foster collaboration while respecting individual responsibilities and maintaining a high standard of quality.



## Branch naming convention(Not Required)

#### 1. **Feature Branches**

These are branches where new features are developed.

* **Format:** `feature/<feature-name>` or `feature/<issue-number>`
* **Example:** `feature/login-system`, `feature/123`

#### 2. **Bugfix Branches**

Branches used for fixing bugs.

* **Format:** `bugfix/<bug-name>` or `bugfix/<issue-number>`
* **Example:** `bugfix/login-error`, `bugfix/456`

#### 3. **Hotfix Branches**

Urgent fixes that need to be applied directly to the production environment.

* **Format:** `hotfix/<issue>` or `hotfix/<issue-number>`
* **Example:** `hotfix/missing-file`, `hotfix/789`

#### 4. **Release Branches**

Used for preparing releases.

* **Format:** `release/<version>` or `release/<release-date>`
* **Example:** `release/1.0.1`, `release/october-2023`

#### 5. **Refactor Branches**

Branches for code refactoring.

* **Format:** `refactor/<description>`
* **Example:** `refactor/cleanup-routing`

#### 6. **Documentation Branches**

For updating or adding documentation.

* **Format:** `docs/<description>` or `docs/<issue-number>`
* **Example:** `docs/readme-update`, `docs/321`

#### 7. **Experimental/Branches for Testing**

Branches for experiments or trials.

* **Format:** `experiment/<description>` or `test/<description>`
* **Example:** `experiment/new-algorithm`, `test/ui-overhaul`

#### 8. **Personal/Developer Branches**

Personal branches for developers, typically for work-in-progress (WIP) code.

* **Format:** `<username>/<description>`
* **Example:** `john/feature-x-work`, `alice/bugfix-y`

#### General Guidelines

* **Be Descriptive and Concise:** The branch name should briefly describe the purpose of the branch.
* **Use Dashes to Separate Words:** Avoid spaces and use dashes (`-`) for readability.
* **Avoid Special Characters:** Stick to alphanumeric characters and dashes.
* **Consider Including Issue/Task Numbers:** If your team uses a ticketing system, including the issue or task number can be helpful for traceability.

Remember, the key to a good naming convention is consistency and clarity. The convention should make it easy for anyone in the team to understand the purpose of a branch at a glance.
