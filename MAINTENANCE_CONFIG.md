# Photo Culling System - Maintenance Configuration

## Automated Maintenance Schedule

### Daily Tasks (Automated)
```bash
# Run maintenance check without cleaning
python tools/project_maintenance.py

# Check for new large files or issues
# Report saved to MAINTENANCE_REPORT.json
```

### Weekly Tasks (Manual)
```bash
# Run full maintenance with cleaning
python tools/project_maintenance.py --clean

# Update dependencies (if needed)
pip list --outdated

# Run integration tests
python tools/integration_test.py
```

### Monthly Tasks (Manual)
```bash
# Full project health check
python tools/health_check_complete.py

# Review and clean git history if needed
git log --oneline -10

# Backup critical databases
cp data/features/*.db backups/
cp data/labels/*.db backups/
```

## Maintenance Thresholds

- **Large Files**: Alert if any file > 10MB
- **Temp Files**: Clean if > 10 temp files
- **__pycache__**: Clean if > 5 directories
- **Database Size**: Monitor if total > 100MB
- **Untracked Files**: Review if > 20 files

## Git Ignore Patterns

The `.gitignore` is configured to exclude:
- ✅ All image files (*.jpg, *.png, etc.)
- ✅ Database files (*.db, *.sqlite)
- ✅ Python cache (__pycache__, *.pyc)
- ✅ Large model files (*.pkl, *.h5, etc.)
- ✅ Temporary files (*.tmp, *.log)
- ✅ IDE files (.vscode/, .idea/)
- ✅ OS files (.DS_Store, Thumbs.db)
- ✅ Environment files (.env, config_local.json)

## Automation Setup (Optional)

### Cron Job for Daily Maintenance
```bash
# Add to crontab (crontab -e)
0 9 * * * cd /path/to/Photo-Culling && python tools/project_maintenance.py
```

### GitHub Actions (Future)
```yaml
# .github/workflows/maintenance.yml
name: Daily Maintenance
on:
  schedule:
    - cron: '0 9 * * *'
jobs:
  maintenance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Maintenance
        run: python tools/project_maintenance.py
```

## Project Health Indicators

### ✅ Good Health
- No large untracked files
- Database sizes reasonable (< 100MB total)
- No accumulated temp files
- Clean git status
- Regular commits following semantic standards

### ⚠️ Attention Needed
- Multiple large files appearing
- Database growth > 10MB/week
- Many untracked files
- Pending commits for > 1 week

### 🚨 Action Required
- Database size > 500MB
- Project size > 1GB
- Many failed maintenance runs
- Git repository issues

## Best Practices Reminder

1. **Never commit large files** (images, models, databases)
2. **Use semantic commits** (feat:, fix:, docs:, etc.)
3. **Regular maintenance** (weekly minimum)
4. **Monitor disk usage** (especially databases)
5. **Keep .gitignore updated** as project evolves
6. **Document any manual cleanup steps**

---

*This configuration is automatically updated by the maintenance system.*
