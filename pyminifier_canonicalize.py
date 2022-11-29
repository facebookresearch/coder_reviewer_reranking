# Copyright (c) Meta Platforms, Inc. and affiliates.

import keyword, sys
from pyminifier import analyze
from pyminifier.minification import remove_comments_and_docstrings, remove_blank_lines
import re

RESERVED_WORDS = keyword.kwlist + analyze.builtins


def clean_comment(code):
    code = remove_comments_and_docstrings(code)
    code = remove_blank_lines(code)
    return code


def remove_print(code):
    code = re.sub("print(.+)", "print('')", code)
    code = re.sub("Error(.+)", "Error('')", code)
    code = re.sub("Exception(.+)", "Exception('')", code)
    code = re.sub("assert (.+), +['\"].+['\"]", "assert \\1", code)
    return code
