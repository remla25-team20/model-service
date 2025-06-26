#!/bin/bash
#
# Returns a new version based on an old version and an (optional) tag.
# If the old version already has a tag, it is copied unless another tag is given.
# If no tag is given and the old version has no tag, defaults to "rc".
# Increments tail of version if old version had a matching tag; otherwise sets tail to 1.
#
# Example inputs:
# 0.5.2-alpha.5  alpha  -->   0.5.2-alpha.6
# 0.5.2-alpha.5         -->   0.5.2-alpha.6
# 0.5.2-alpha.5  beta   -->   0.5.2-beta.1
#
# 2.8.9          gamma  -->   2.8.10-gamma.1
# 2.8.9-alpha.5  gamma  -->   2.8.9-gamma.1
# 2.8.9                 -->   2.8.10-rc.1
#
# Arguments:
#   $1 - old version
#   $2 - tag (optional) - branch name, or "rc"

old_version=$1

version_base=${old_version%%-*}
version_tail=${old_version##*.}
((version_tail++))
version_tag=${old_version#*-}
version_tag=${version_tag%.*}

release_tag=${2:-$version_tag}

if [[ "$old_version" != *"-"* ]]; then 
	release_tag=${2:-rc}
	version_base_base=${version_base%.*}
	version_base_tail=${version_base##*.}
	((version_base_tail++))
	version_base="${version_base_base}.${version_base_tail}"
fi

if [[ "$version_tag" != "$release_tag" ]]; then 
	version_tail=1
fi;

echo "${version_base}-${release_tag}.${version_tail}"
