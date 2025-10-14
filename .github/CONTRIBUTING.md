# 🤝 Contributing to DeepFakeDetector
*A project by the McMaster Artificial Intelligence Society (Fall 2025)*

---

## 📋 Getting Started
Welcome to the **DeepFakeDetector** project! We're excited to have you contribute to our AI-generated image detection research. Please follow these guidelines to ensure a smooth collaboration process.

---

## 🌿 Branching Strategy
**Important:** Always branch off the `development` branch, never directly from `main`.

### Branch Naming Convention
Use the following prefixes based on the type of work you're doing:

- **`model/`** - For machine learning model development, training, or architecture changes
  - Example: `model/cnn-implementation`, `model/vit-optimization`
- **`data/`** - For dataset management, preprocessing, or data pipeline work
  - Example: `data/preprocessing-pipeline`, `data/augmentation-strategy`
- **`web/`** - For frontend dashboard or web interface development
  - Example: `web/upload-component`, `web/results-visualization`
- **`api/`** - For backend API development or FastAPI endpoints
  - Example: `api/prediction-endpoint`, `api/image-processing`
- **`research/`** - For experimental features or research exploration
  - Example: `research/fft-analysis`, `research/explainability-methods`
- **`docs/`** - For documentation updates or technical writing
  - Example: `docs/model-architecture`, `docs/deployment-guide`
- **`fix/`** - For bug fixes across any component
  - Example: `fix/memory-leak`, `fix/prediction-accuracy`

### Creating Your Branch
1. **Switch to development branch:**
   ```bash
   git checkout development
   ```

2. **Pull the latest changes:**
   ```bash
   git pull origin development
   ```

3. **Create your feature branch:**
   ```bash
   git checkout -b <prefix>/<descriptive-name>
   ```
   
   Example: `git checkout -b model/hybrid-cnn-vit`

## 🚀 Pushing Changes
After you've committed your changes and tested your work, you're ready to push to GitHub.

If you created your branch locally, you'll need to set the upstream the first time you push:

```bash
git push --set-upstream origin <branch-name>
```

This command pushes your branch to GitHub and tells Git to track the remote version of your branch, so future pushes can be done with just:

```bash
git push
```

To confirm which branch you're currently on before pushing, you can use:

```bash
git branch
```

The active branch will be marked with an asterisk (*).

---

## 📝 Making a Pull Request

When you are finished with your task and have tested it successfully, you can create a new PR:

1. Visit the **Pull Requests** tab and click the green **New pull request** button

2. **Select the development branch as the base branch**
3. **Select your own branch as the compare branch**

4. Once it has verified that the branches can be merged, click **Create pull request**. If it says "Can't automatically merge", you may need to rebase your branch with the latest changes from `development`

5. Your PR will come with a template to fill in:
   - Issue number (if applicable)
   - Description of changes made
   - Screenshots (for web/UI changes)
   - Test results (for model/data changes)
   - Checklist to verify your changes

6. **Add yourself as the assignee** to the PR
7. **Add your team lead as your reviewer** to the PR

8. Once you're done, click **Create pull request**!