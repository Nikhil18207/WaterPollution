# Git Commands to Push Your Water Pollution Project

## Step-by-Step Instructions

### 1. Open Git Bash in your project directory
Navigate to: `E:\ML\Projects\WaterPollutionProject`

### 2. Initialize Git Repository (if not already initialized)
```bash
git init
```

### 3. Configure Git (if first time using Git)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 4. Add the remote repository
```bash
git remote add origin https://github.com/Nikhil18207/WaterPollution.git
```

### 5. Check current status
```bash
git status
```

### 6. Add all files to staging
```bash
git add .
```

### 7. Commit your changes
```bash
git commit -m "Initial commit: Water Pollution Project"
```

### 8. Push to GitHub
```bash
git push -u origin main
```

**Note:** If the default branch is `master` instead of `main`, use:
```bash
git push -u origin master
```

---

## Alternative: If repository already exists on GitHub

If you've already created files on GitHub, you might need to pull first:

```bash
git pull origin main --allow-unrelated-histories
```

Then push:
```bash
git push -u origin main
```

---

## Common Issues and Solutions

### Issue 1: Branch name mismatch
If you get an error about branch names, check your current branch:
```bash
git branch
```

Rename branch to main if needed:
```bash
git branch -M main
```

### Issue 2: Authentication required
GitHub requires personal access token for HTTPS. You have two options:

**Option A: Use Personal Access Token**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with 'repo' scope
3. Use token as password when prompted

**Option B: Use SSH (Recommended)**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add SSH key to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key and add to GitHub
cat ~/.ssh/id_ed25519.pub
```

Then change remote URL to SSH:
```bash
git remote set-url origin git@github.com:Nikhil18207/WaterPollution.git
```

### Issue 3: Large files
If you have large model files, consider using Git LFS:
```bash
git lfs install
git lfs track "*.pkl"
git lfs track "*.pth"
git add .gitattributes
```

---

## Quick Command Summary

```bash
# Full workflow in one go:
cd /e/ML/Projects/WaterPollutionProject
git init
git add .
git commit -m "Initial commit: Water Pollution Project"
git branch -M main
git remote add origin https://github.com/Nikhil18207/WaterPollution.git
git push -u origin main
```

---

## Verify Your Push

After pushing, visit: https://github.com/Nikhil18207/WaterPollution

You should see all your files there!

---

## Future Updates

After the initial push, for future updates use:

```bash
git add .
git commit -m "Description of changes"
git push
```