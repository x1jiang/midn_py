# PDF Generation Notes

The PLAYBOOK.pdf file needs to be regenerated when PLAYBOOK.md is updated.

## Option 1: Using Pandoc with XeLaTeX (handles Unicode)

```bash
cd vantage6_algorithms
pandoc PLAYBOOK.md -o PLAYBOOK.pdf --pdf-engine=xelatex -V geometry:margin=1in
```

## Option 2: Using Pandoc with LaTeX (may need emoji removal)

If XeLaTeX is not available, you may need to remove emoji characters (✅, 🔄, etc.) from the markdown first, or use:

```bash
pandoc PLAYBOOK.md -o PLAYBOOK.pdf --pdf-engine=pdflatex -V geometry:margin=1in
```

## Option 3: Online Tools

- Use online markdown to PDF converters
- Use GitHub's PDF export feature
- Use VS Code extensions like "Markdown PDF"

## Current Status

- Markdown source (PLAYBOOK.md) is up to date ✅
- PDF may be outdated if regenerated with standard LaTeX
- Recommend using XeLaTeX for Unicode support

---
Last updated: 2025-11-27
