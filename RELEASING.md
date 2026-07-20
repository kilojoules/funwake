# Releasing FunWake (DOI-minted snapshot for the paper)

The paper's Appendix A ends with "archived at [TODO]". That link should be a
Zenodo DOI pointing at a tagged snapshot of this repository. Minting a DOI is a
**human step** — GitHub↔Zenodo cannot be enabled from CI.

## One-time setup (human)

1. Sign in to https://zenodo.org with the GitHub account that owns
   `kilojoules/funwake`.
2. In Zenodo → **Settings → GitHub**, flip the toggle **on** for the
   `kilojoules/funwake` repository. (Zenodo only archives releases created
   *after* the toggle is enabled.)
3. `.zenodo.json` in the repo root supplies the archive metadata (title,
   authors, license, keywords). Review it and fill any `TODO(human)` fields in
   `CITATION.cff` (affiliations, ORCIDs) first.

## Cutting the release (human, after CI is green)

1. Confirm the `smoke` workflow is green on `main` (badge in the README).
2. Tag and push:

   ```bash
   git tag -a v0.1-icml2026 -m "ICML 2026 submission snapshot"
   git push origin v0.1-icml2026
   ```

3. On GitHub, **Releases → Draft a new release → choose tag `v0.1-icml2026`**,
   title it, and **Publish**. Zenodo then archives the tarball and mints a DOI.
4. Copy the DOI badge/URL from Zenodo into:
   - `CITATION.cff` (`doi:` and `version:` fields), and
   - the paper's Appendix A "archived at ..." line.

## Notes

- Tagging is intentionally left to the human so the snapshot corresponds to a
  reviewed state, not an automated commit.
- Re-running the release for a later version mints a *new* version DOI under the
  same concept DOI.
