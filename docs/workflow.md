# GitHub Flow

This project follows the GitHub Flow model, which consists of the following steps:

## Workflow Steps

1. **Create a branch**: Branch off from `main` with a descriptive name (e.g., `feature/add-login`).
2. **Make changes**: Commit logical, atomic changes with clear messages.
3. **Open a Pull Request**: Push your branch and open a PR against `main`.
4. **Collaborate and review**: Request reviews, discuss changes, and address feedback.
5. **Ensure CI passes**: Wait for automated tests and lint checks to succeed.
6. **Merge the Pull Request**: Once approved, merge into `main` (using merge commit or squash merge).
7. **Deploy**: Deploy the updated `main` branch to the production environment.
8. **Clean up**: Delete the feature branch locally and remotely after merging.

## Creating a Branch

```bash
# Ensure you have the latest main branch
git checkout main
git pull origin main

# Create a new feature branch
git checkout -b feature/new-feature
```

## Making Changes

```bash
# Make your changes and commit them
git add .
git commit -m "feat: add new feature"

# Push your branch to GitHub
git push -u origin feature/new-feature
```

## Opening a Pull Request

1. Go to the repository on GitHub
2. Click "Pull requests" > "New pull request"
3. Select your branch as the compare branch
4. Fill in the PR template with details about your changes
5. Submit the pull request

## After Merge

```bash
# Switch back to main
git checkout main
git pull origin main

# Delete the local branch
git branch -d feature/new-feature

# Delete the remote branch (optional)
git push origin --delete feature/new-feature
```
