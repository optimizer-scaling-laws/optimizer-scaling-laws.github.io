# Blog launch package

This package is structured for a Jekyll/GitHub Pages `docs/` directory.

Key files:
- `_posts/2026-06-18-architecture-creates-capacity-optimizers-realize-it.md` — main blog post.
- `_layouts/default.html`, `_layouts/post.html` — minimal layouts with MathJax support.
- `assets/css/blog.css` — blog styling.
- `assets/blog/architecture-optimizer-codesign/` — publication figures used by the post.

Figures included:
1. `figure1_optimizer_capacity_scaling.png`
2. `figure2_matched_loss_capacity_comparison.png` / `.svg`
3. `figure3_phase_diagram_for_realized_capacity.png`
4. `figure4_capacity_scaling_asymmetry_token_regimes.png`

The post uses `relative_url` paths, so it should work under GitHub Pages project-site paths.
