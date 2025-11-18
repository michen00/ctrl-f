#!/usr/bin/env bash

set -e

# Parse command line arguments
JSON_MODE=false
ARGS=()

for arg in "$@"; do
  case "$arg" in
    --json)
      JSON_MODE=true
      ;;
    --help | -h)
      echo "Usage: $0 [--json]"
      echo "  --json    Output results in JSON format"
      echo "  --help    Show this help message"
      exit 0
      ;;
    *)
      ARGS+=("$arg")
      ;;
  esac
done

# Get script directory and load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.specify/scripts/bash/common.sh
source "$SCRIPT_DIR/common.sh"

# Get all paths and variables from common functions
# Variables are set by eval from get_feature_paths output
eval "$(get_feature_paths)"

# Check if we're on a proper feature branch (only for git repos)
# shellcheck disable=SC2153
check_feature_branch "$CURRENT_BRANCH" "$HAS_GIT" || exit 1

# Ensure the feature directory exists
# shellcheck disable=SC2153
mkdir -p "$FEATURE_DIR"

# Copy plan template if it exists
# shellcheck disable=SC2153
TEMPLATE="$REPO_ROOT/.specify/templates/plan-template.md"
if [[ -f $TEMPLATE ]]; then
  # shellcheck disable=SC2153
  cp "$TEMPLATE" "$IMPL_PLAN"
  # shellcheck disable=SC2153
  echo "Copied plan template to $IMPL_PLAN"
else
  echo "Warning: Plan template not found at $TEMPLATE"
  # Create a basic plan file if template doesn't exist
  touch "$IMPL_PLAN"
fi

# Output results
if $JSON_MODE; then
  # shellcheck disable=SC2153
  printf '{"FEATURE_SPEC":"%s","IMPL_PLAN":"%s","SPECS_DIR":"%s","BRANCH":"%s","HAS_GIT":"%s"}\n' \
    "$FEATURE_SPEC" "$IMPL_PLAN" "$FEATURE_DIR" "$CURRENT_BRANCH" "$HAS_GIT"
else
  # shellcheck disable=SC2153
  echo "FEATURE_SPEC: $FEATURE_SPEC"
  # shellcheck disable=SC2153
  echo "IMPL_PLAN: $IMPL_PLAN"
  # shellcheck disable=SC2153
  echo "SPECS_DIR: $FEATURE_DIR"
  # shellcheck disable=SC2153
  echo "BRANCH: $CURRENT_BRANCH"
  # shellcheck disable=SC2153
  echo "HAS_GIT: $HAS_GIT"
fi
