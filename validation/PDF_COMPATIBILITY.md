# PDF compatibility fix — October 8, 2026

The supplied 2024 CU Answers report reproduced `invalid_pdf` before extraction. The underlying exception was pypdf's missing AES crypto dependency. After installing the dependency, the PDF's empty opening password successfully decrypted it; the existing blanket `is_encrypted` rejection was a second independent bug.

The app now installs `pypdf[crypto]` through the project manifest and lockfile. It accepts encrypted PDFs only when an empty opening password succeeds, and continues to reject PDFs requiring a password. Missing crypto support returns a distinct sanitized server-configuration error instead of suggesting the PDF is scanned. The uploaded original is neither modified nor copied into this repository.

Local extraction verified 77 text-bearing pages and 170,112 extracted characters before trimming. Full multipart endpoint verification uses the real file with a mock AI provider, so it checks upload, worker-process parsing, source locations, and full-context preflight without sending the report to an external model. It is not an answer-quality test.

90 automated tests passed, including AES PDFs with and without an opening password, page-limit enforcement on encrypted PDFs, and the missing-dependency error. Python lint and formatting passed.

Dependency behavior reference: https://github.com/py-pdf/pypdf/blob/main/docs/user/encryption-decryption.md
