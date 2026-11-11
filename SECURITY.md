# Security Policy

## Reporting a vulnerability

To report a security issue, please use https://g.co/vulnz. We use g.co/vulnz
for our intake, and do coordination and disclosure here on GitHub (including
using GitHub Security Advisory). The Google Security Team will respond within
5 working days of your report on g.co/vulnz.

## Security model

Ginger is a library for building and searching vector indexes. It assumes that
datasets, configs and index files come from a trusted source.

Loading an index checks the file for corruption and truncation, but the loader
is not hardened against maliciously crafted files. Only load index files that
you or another trusted party wrote.
