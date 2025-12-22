# Proxide Configuration System

## Goals

Implement a global configuration system for `proxide` using a `proxide.ini` file in the base directory or user home directory.

## Features

- **Fetch Limits**: Configurable limit for direct downloads vs database queries (e.g., AFDB direct vs FoldComp).
  - Variable: `AFDB_FETCH_LIMIT`
  - Default: 50
- **Cache Paths**: Configurable locations for caching downloaded structures.
- **Proxy Settings**: HTTP proxy configuration for `reqwest`.

## Implementation Tasks

- [ ] Create `proxide.config` module.
- [ ] Support loading from `~/.config/proxide.ini` and `./proxide.ini`.
- [ ] Integrate with Rust backend (pass config struct to Rust).
