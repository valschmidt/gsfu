#!/usr/bin/env bash
#
# Copyright (c) 2026, Val Schmidt, University of New Hampshire
# SPDX-License-Identifier: BSD-2-Clause
# See the LICENSE file at the top of this repository for the full text.
#
# Build gsfu and upload it to PyPI, then tag the release in git. With
# --test, upload to TestPyPI instead, then install the upload into a fresh
# virtual environment and test it there.
# Run ./release.sh -h for usage and the full release procedure.

# Treat unset variables and failures anywhere in a pipeline as errors.
# Not -e: each step checks its own result and calls fail() with a message.
set -uo pipefail

usage() {
    cat <<'EOF'
Usage: ./release.sh [--test [--skip-upload] | -h]

Build gsfu and upload it to PyPI, then tag the release in git.

Options:
  (none)         Run the tests, build, check, upload to PyPI, then create
                 the tag vX.Y.Z and push main and the tag to origin.
  --test         Run the tests, build, check, and upload to TestPyPI; then
                 install that upload into a fresh virtual environment and
                 test the installed package there (see below). No tag.
  --test --skip-upload
                 Don't build or upload; rerun the install-and-test checks
                 on the version already on TestPyPI that matches setup.py.
  -h, --help     Show this help and exit.

The version released is whatever setup.py declares. The script must be
run on the main branch, refuses to run with uncommitted changes or when
that version is already on the index it would upload to, and asks for
confirmation before uploading anything.

After a TestPyPI upload, --test installs gsfu[kmall] (gsfu plus the
optional pykmall dependency kmall2gsf.py needs) and checks the installed
package, not the source tree: that GSFU imports from the fresh
environment at the right version, that KMALL imports, that the gsfu.py
and kmall2gsf.py commands run, that gsfu.py -V indexes a sample file
(if data/GSF/ exists), and that the full test suite passes against the
installed package, including kmall2gsf's real-file conversion tests
when their sample .kmall files are present.

Release procedure:
  1. Make changes on main and commit them; repeat as needed.
  2. Edit the version in setup.py to X.Y.Z, and commit that change on
     its own. Do not tag it; this script creates the tag.
  3. ./release.sh --test
  4. Optionally, look over the project page on test.pypi.org.
  5. ./release.sh

If step 3 fails after uploading, fix the problem and commit. TestPyPI
never accepts the same version twice, so test the fix under a
pre-release version: set setup.py to X.Y.Zrc1 (then rc2, and so on),
commit, and repeat step 3 with that version. Once it passes, set
setup.py back to X.Y.Z, commit, and run step 5. TestPyPI and PyPI are
separate indexes, so X.Y.Z having been used on TestPyPI does not stop
it going to PyPI.

Environment:
  PYTHON   The interpreter used to create the virtual environments and
           run the source-tree tests (default: python3). It needs pytest,
           pandas, and numpy installed.

Credentials: uploads run non-interactively, so they must be configured
beforehand, either in ~/.pypirc ([pypi] and [testpypi] sections, listed
under index-servers, each with username __token__ and an API token as
the password), or in the TWINE_USERNAME and TWINE_PASSWORD environment
variables.

On success the temporary work directory is removed. On failure it is
kept, and its location printed, so the log, build, and environments can
be inspected.
EOF
}

# --- Arguments -------------------------------------------------------------

# Choose the upload target: PyPI by default, TestPyPI with --test.
repository=pypi
skip_upload=0
case "${1:-}" in
    "")        ;;
    --test)    repository=testpypi ;;
    -h|--help) usage; exit 0 ;;
    *)         usage >&2; exit 2 ;;
esac
case "${2:-}" in
    "")            ;;
    --skip-upload) [[ "$repository" == testpypi ]] || { usage >&2; exit 2; }
                   skip_upload=1 ;;
    *)             usage >&2; exit 2 ;;
esac
[[ $# -le 2 ]] || { usage >&2; exit 2; }

PYTHON="${PYTHON:-python3}"

# Run from the repository root, wherever the script is invoked from.
REPO="$(cd "$(dirname "$0")" && pwd -P)"
cd "$REPO" || exit 1

# --- Helpers ---------------------------------------------------------------

# Every step's output goes to a log in a temporary work directory, which is
# kept on failure. Resolve its path fully (macOS TMPDIR ends in "/" and /var
# is a symlink to /private/var) so it compares cleanly with the paths
# Python reports.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/gsfu-release.XXXXXX")" || { echo "FAIL: mktemp failed" >&2; exit 1; }
WORK="$(cd "$WORK" && pwd -P)"
LOG="$WORK/log.txt"
TOOLS="$WORK/tools-venv"
TESTENV="$WORK/test-venv"
DIST="$WORK/dist"

step() {
    echo "==> $1"
    echo "==> $1" >> "$LOG"
}

fail() {
    echo
    echo "FAIL: $1"
    # A failure before any step ran (a pre-flight check) leaves nothing
    # worth keeping.
    if [[ ! -s "$LOG" ]]; then
        rm -rf "$WORK"
        exit 1
    fi
    echo "  Last lines of the log:"
    tail -n 20 "$LOG" | sed 's/^/    /'
    echo
    echo "  Full log:        $LOG"
    echo "  Kept for review: $WORK"
    exit 1
}

# Run a command, sending its output to the log.
run() {
    "$@" >> "$LOG" 2>&1
}

# --- Checks before anything is built ---------------------------------------

# setup.py is the single source of the package name and version.
NAME="$(sed -nE 's/^[[:space:]]*name[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' setup.py)"
VERSION="$(sed -nE 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' setup.py)"
[[ -n "$NAME" && -n "$VERSION" ]] || fail "could not read name and version from setup.py"
TAG="v$VERSION"

if [[ "$skip_upload" -eq 1 ]]; then
    echo "Testing $NAME $VERSION already on TestPyPI (work dir: $WORK)"
else
    echo "Releasing $NAME $VERSION to $repository (work dir: $WORK)"
fi

# The tag is created on the checked-out commit but only main is pushed,
# so releasing from any other branch would tag a commit not on main.
branch="$(git branch --show-current)"
[[ "$branch" == main ]] || fail "on branch '$branch'; releases are made from main"

# Only release committed code, so the uploaded package matches the tag.
[[ -z "$(git status --porcelain)" ]] || fail "working tree is not clean; commit or stash first"

# PyPI never accepts the same version twice, so an existing tag means the
# version in setup.py has not been bumped since the last release.
if [[ "$repository" == pypi ]] && git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    fail "tag $TAG already exists; bump the version in setup.py first"
fi

# Neither index ever accepts the same version twice. A repeated upload is
# rejected only after the tests and build, and the index's error doesn't
# always say why, so ask the index first. If it can't be reached, carry on
# and let the upload report the problem.
if [[ "$skip_upload" -eq 0 ]]; then
    if [[ "$repository" == testpypi ]]; then index_host=test.pypi.org; else index_host=pypi.org; fi
    status="$(curl -s -o /dev/null -w '%{http_code}' "https://$index_host/pypi/$NAME/$VERSION/json")"
    if [[ "$status" == 200 && "$repository" == testpypi ]]; then
        fail "$NAME $VERSION is already on TestPyPI. Bump the version in setup.py, or rerun with --test --skip-upload to test the existing upload"
    elif [[ "$status" == 200 ]]; then
        fail "$NAME $VERSION is already on PyPI. Bump the version in setup.py first"
    fi
fi

# The upload must be confirmed at a prompt, which needs a terminal to read
# from. Without one the answer would be empty and read as "no", so stop
# now rather than after the tests and build.
if [[ "$skip_upload" -eq 0 && ! -t 0 ]]; then
    fail "no terminal to confirm the upload from; run this from a terminal"
fi

# --- Build and upload ------------------------------------------------------

if [[ "$skip_upload" -eq 0 ]]; then
    step "Running the tests against the source tree"
    run "$PYTHON" -m pytest -q -p no:cacheprovider tests || fail "tests failed against the source tree"

    # Build and upload with tools from their own environment, so the result
    # doesn't depend on whatever happens to be installed in $PYTHON.
    step "Creating build tools environment"
    run "$PYTHON" -m venv "$TOOLS" || fail "could not create build tools venv with $PYTHON"
    run "$TOOLS/bin/pip" install --upgrade pip build twine || fail "could not install build and twine"

    # Build into the work directory, so no stale files from an earlier
    # build in the repository's dist/ can be uploaded by mistake.
    step "Building sdist and wheel"
    run "$TOOLS/bin/python" -m build --outdir "$DIST" "$REPO" || fail "package build failed"

    # Validate the package metadata, including that the README renders.
    run "$TOOLS/bin/twine" check "$DIST"/* || fail "twine check failed on the built files"

    # Uploads cannot be undone, so confirm before sending anything.
    echo "    Built: $(cd "$DIST" && echo *)"
    read -r -p "Upload $NAME $VERSION to $repository? [y/N] " answer
    [[ "$answer" == [yY] ]] || fail "upload declined"

    step "Uploading to $repository"
    if ! run "$TOOLS/bin/twine" upload --non-interactive --repository "$repository" "$DIST"/*; then
        if grep -qi "File already exists" "$LOG"; then
            fail "$NAME $VERSION is already on $repository. Bump the version in setup.py$( \
                [[ "$repository" == testpypi ]] && echo ", or rerun with --test --skip-upload to test the existing upload")"
        fi
        fail "upload to $repository failed (check the credentials described in -h)"
    fi
fi

# --- Real release: tag and push --------------------------------------------

if [[ "$repository" == pypi ]]; then
    step "Tagging $TAG and pushing main and the tag to origin"
    run git tag "$TAG" || fail "$NAME $VERSION is on PyPI, but creating tag $TAG failed; finish with: git tag $TAG && git push origin main $TAG"
    run git push origin main "$TAG" \
        || fail "$NAME $VERSION is on PyPI, but the push failed; finish with: git push origin main $TAG"
    rm -rf "$WORK"
    echo
    echo "PASS: released $NAME $VERSION to PyPI and pushed tag $TAG."
    exit 0
fi

# --- TestPyPI: install the upload into a fresh environment and test it -----

# TestPyPI can take a little while before a new release can be installed.
step "Waiting for $NAME $VERSION to appear on TestPyPI"
for i in $(seq 1 24); do
    curl -sf -o /dev/null "https://test.pypi.org/pypi/$NAME/$VERSION/json" && break
    [[ "$i" -eq 24 ]] && fail "$NAME $VERSION did not appear on TestPyPI within 2 minutes"
    sleep 5
done

step "Creating test environment"
run "$PYTHON" -m venv "$TESTENV" || fail "could not create test venv with $PYTHON"

# Install with the kmall extra, so the pykmall dependency declared in
# setup.py is tested too. Dependencies (pandas, numpy, pykmall) come from
# the real PyPI, since TestPyPI's copies are often missing or stale. The
# simple index can lag the JSON API checked above, so retry for a while.
step "Installing $NAME[kmall]==$VERSION from TestPyPI"
installed=0
for i in 1 2 3 4 5 6; do
    if run "$TESTENV/bin/pip" install --no-cache-dir \
            -i https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            "$NAME[kmall]==$VERSION"; then
        installed=1
        break
    fi
    sleep 10
done
[[ "$installed" -eq 1 ]] || fail "could not install $NAME[kmall]==$VERSION from TestPyPI"

# Installed separately, after gsfu, so pytest can't affect which gsfu
# version or dependencies get resolved.
run "$TESTENV/bin/pip" install pytest || fail "could not install pytest in the test environment"

# Work from inside the work directory from here on, so the GSFU package in
# the repository can't be imported by mistake instead of the installed one.
cd "$WORK" || fail "could not change to $WORK"

step "Checking the installed package is the TestPyPI one"
LOCATION="$("$TESTENV/bin/python" -c 'import GSFU, os; print(os.path.realpath(os.path.dirname(GSFU.__file__)))' 2>> "$LOG")" \
    || fail "'import GSFU' failed in the test environment"
case "$LOCATION" in
    "$TESTENV"/*) ;;
    *) fail "GSFU was imported from $LOCATION, not from the test environment" ;;
esac
GOT="$("$TESTENV/bin/pip" show "$NAME" 2>> "$LOG" | sed -n 's/^Version: //p')"
[[ "$GOT" == "$VERSION" ]] || fail "installed $NAME version is '$GOT', expected '$VERSION'"

# Without this, the kmall2gsf real-file tests would silently skip.
run "$TESTENV/bin/python" -c 'import KMALL.kmall' \
    || fail "'import KMALL.kmall' failed; the kmall extra did not install pykmall"

# The console-script entry points declared in setup.py.
step "Running 'gsfu.py -h' and 'kmall2gsf.py -h'"
run "$TESTENV/bin/gsfu.py" -h || fail "'gsfu.py -h' failed"
run "$TESTENV/bin/kmall2gsf.py" -h || fail "'kmall2gsf.py -h' failed"

SAMPLE="$(ls -S "$REPO"/data/GSF/*.gsf 2>/dev/null | tail -n 1)"
if [[ -n "$SAMPLE" ]]; then
    step "Indexing $(basename "$SAMPLE") with 'gsfu.py -V'"
    run "$TESTENV/bin/gsfu.py" -f "$SAMPLE" -V || fail "'gsfu.py -V' failed on $SAMPLE"
else
    step "Skipping sample-file check: no files in $REPO/data/GSF/"
fi

# Run the repository's own tests against the installed package. The tests
# are copied (not linked) into the work directory, so they find sample data
# through a data/ link next to them, exactly as they do in the repository,
# while GSFU itself can only come from the test environment.
step "Running the test suite against the installed package"
mkdir -p "$WORK/suite"
cp -R "$REPO/tests" "$WORK/suite/tests" || fail "could not copy the tests"
rm -rf "$WORK/suite/tests/__pycache__"
[[ -d "$REPO/data" ]] && ln -s "$REPO/data" "$WORK/suite/data"
(cd "$WORK/suite" && run "$TESTENV/bin/python" -m pytest -q -p no:cacheprovider tests) \
    || fail "tests failed against the installed package"
echo "    $(tail -n 1 "$LOG")"

cd "$REPO" || true
rm -rf "$WORK"
echo
echo "PASS: $NAME $VERSION installs from TestPyPI and passes its tests. Temporary files removed."
