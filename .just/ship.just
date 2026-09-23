# Pushing a vX.Y.Z tag runs .github/workflows/release.yml, which publishes the
# wheel to PyPI, the crates to crates.io, and the CLI tarball to a GitHub release.

# Bump the version (patch/minor/major), commit, tag and push to trigger a release
release bump_level="patch":
    #!/usr/bin/env -S echo-comment --shell-flags="-euo pipefail" --color bright-green

    ## Exit early if help was requested
    if [[ "{{bump_level}}" == "--help" ]]; then
        cargo set-version --help
        exit 0
    fi

    # 🔍 Refuse to release from a dirty tree, or from anything but an up-to-date master
    if [[ -n "$(git status --porcelain)" ]]; then
        git status --short
        exit 1
    fi
    branch_name=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$branch_name" != "master" ]]; then
        # ❌ Not on master (current: $branch_name)
        exit 1
    fi
    git fetch origin master
    if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master)" ]]; then
        # ❌ Local master differs from origin/master; pull first
        exit 1
    fi

    # ✅ Pre-release checks
    just check

    # 📈 Bump the workspace version ({{bump_level}}): every crate and the wheel
    cargo set-version --bump {{bump_level}}
    new_version=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.name == "cujson") | .version')

    # 📦 Commit the bump and tag it
    git add --all
    git commit -m "chore(release): bump -> v$new_version"
    git tag -a "v$new_version" -m "Release $new_version"

    # 🚀 Push the release commit, then the tag (the tag triggers release.yml)
    git push origin "$branch_name"
    git push origin "v$new_version"
