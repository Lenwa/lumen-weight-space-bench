# License review before publication

The proposed root license is MIT, using the neutral notice "Lumen contributors".
Confirm that attribution wording before publication. This license applies only
to contributions for which the licensor can grant the relevant rights; it does
not establish that every AI-assisted element is independently copyrightable.

MIT permits use, modification, redistribution and commercial reuse while requiring
preservation of its copyright and permission notice. It does not transfer ownership.
The owner may continue using and licensing their own contributions, but cannot
withdraw permissions already granted under MIT for distributed versions. MIT does
not require downstream modifications to be published or paid back. If retaining
exclusive commercial control was intended, a permissive open-source license does
not provide that restriction; resolve that choice before publishing this candidate.

Third-party scripts retain their own MIT notices and full original LICENSE.
Upstream logs/data/networks have separate CC BY 4.0 terms; quoted MinAtar code
has GPL-3.0 terms and quoted benchmarks may have other licenses. None of those
datasets, model weights or MinAtar source is included. Model checkpoints and
separately installed dependencies retain their own licenses.

Sources checked for this preparation:
- https://choosealicense.com/licenses/mit/
- https://choosealicense.com/licenses/apache-2.0/ (alternative with express patent provisions)
- https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE

The retained source audit pins the upstream revision and downloaded file hashes.
A fresh direct network byte re-fetch was unavailable during packaging. Local
vendor bytes must match that retained audit; do not describe this as an independent
fresh upstream-download verification. Publication requires owner review of this
license choice, audit report and exact staged file list. No remote or public push
is configured by this handoff.
