# Security

## Reporting a vulnerability

Use this repository's **Security → Report a vulnerability** page. Do not open a public issue.

Include the commit or build, macOS version, reproduction steps and observed impact. Remove keys, private paths, prompts and conversations from attachments.

## Scope

tokspur serves a local API protected by a generated key. Report problems such as:

- API access without the key, or another user reading private keys, settings or state
- adoption or configuration accepting changed, substituted or symlinked model files, or writing outside `.tokspur/`
- outbound requests from the server, or prompts and responses leaking into logs or shared state
- workers left running after `tokspur serve` stops, or the supervisor stopping unrelated processes
- an installer or update accepting a release that fails checksum, Developer ID signature or build checks

Client coding agents choose which files to read and commands to run. Report problems in those choices to the client project.

## Supported versions

Fixes land on `main`. There are no maintained release branches yet.
